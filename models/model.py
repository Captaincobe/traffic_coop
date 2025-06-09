import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoModel
from utils.loss import kl_divergence_loss_from_logits
from torch.utils.data import DataLoader

PROMPT_LENGTH = 20


class SharedEncoderClassifier(nn.Module):
    def __init__(self, shared_encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = shared_encoder
        self.classifier = nn.Linear(hidden_dim, num_classes).to(torch.float32)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, cls_token=None):
        if cls_token is not None:
            return self.classifier(cls_token)
        if inputs_embeds is not None:
            output = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        return self.classifier(cls_token)
    
class SharedRoBERTaPromptLearner(nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embedding = nn.Parameter(torch.empty(prompt_length, embedding_dim, dtype=torch.float32))
        # nn.init.normal_(self.prompt_embedding, std=0.005)
        nn.init.xavier_uniform_(self.prompt_embedding)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        # prompt_embed = self.prompt_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_embed = self.ln(self.prompt_embedding).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_embed, input_embeds], dim=1)


class PromptLearnerManager(nn.Module):
    def __init__(self, shared_encoder, num_labels, prompt_length, K=4, device="cuda"):
        super().__init__()
        self.device = device
        self.K = K
        self.prompt_length = prompt_length
        self.encoder = shared_encoder
        self.embedding_dim = self.encoder.config.hidden_size

        self.ood_prompts = nn.ModuleList([
            SharedRoBERTaPromptLearner(prompt_length, self.embedding_dim) for _ in range(K)
        ])
        self.coop_prompts = nn.ModuleList([
            SharedRoBERTaPromptLearner(prompt_length, self.embedding_dim) for _ in range(K)
        ])
        self.classifier = nn.Linear(self.embedding_dim, num_labels).to(torch.float32)

    def _forward_with_prompt(self, prompt_module, input_ids, attention_mask):

        batch_size = input_ids.size(0)
        input_embeds = self.encoder.embeddings(input_ids)
        full_embeds = prompt_module(input_embeds)
        full_embeds = full_embeds.to(torch.float32)
        prompt_mask = torch.ones((batch_size, self.prompt_length), dtype=attention_mask.dtype).to(attention_mask.device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        # logits = torch.clamp(logits, min=-10, max=10)
        return logits
        # return self.classifier(cls_token)

    def forward_ood(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.ood_prompts[k], input_ids, attention_mask)

    def forward_coop(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.coop_prompts[k], input_ids, attention_mask)

    def get_all_ood_parameters(self):
        return list(self.ood_prompts.parameters()) + list(self.classifier.parameters())

    def get_all_coop_parameters(self):
        return list(self.coop_prompts.parameters()) + list(self.classifier.parameters())
    
def train_prompt_learner(model, forward_fn, dataloader, optimizer, loss_fn, scheduler=None, max_epochs=3, device="cuda"):
    model.train()
    for epoch in range(max_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = forward_fn(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler: scheduler.step()

            total_loss += loss.item()
        print(f"[PromptTuning][Epoch {epoch+1}/{max_epochs}] Loss: {total_loss/len(dataloader):.4f}")

class LLMTrafficDECOOP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.llm_model_name = args.LLM_MODEL_NAME
        self.device = args.DEVICE
        self.k_detectors = args.K_DETECTORS

        # -------------------- Shared Encoder (frozen) --------------------
        self.shared_encoder = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        self.hidden_dim = self.shared_encoder.config.hidden_size

        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES

        # -------------------- Classifiers --------------------
        ## Zero-shot classifier (frozen)
        self.zs_classifier_ = SharedEncoderClassifier(
            self.shared_encoder, self.hidden_dim, self.num_base_classes
        ).to(self.device)

        ## Sub-classifiers (K detectors)
        self.sub_classifiers_ = nn.ModuleList([
            SharedEncoderClassifier(self.shared_encoder, self.hidden_dim, self.num_base_classes).to(self.device)
            for _ in range(self.k_detectors)
        ])

        # -------------------- Prompt Learner Manager (OOD + COOP prompts) --------------------
        self.prompt_manager = PromptLearnerManager(
            shared_encoder=self.shared_encoder,
            num_labels=self.num_base_classes,
            prompt_length=PROMPT_LENGTH,
            K=self.k_detectors,
            device=self.device
        ).to(self.device)

        self.cp_entropy_scores = []
        self.eci_thresholds = []

    def set_base_class_global_indices(self, base_class_global_indices):
        self.base_class_global_indices = base_class_global_indices
    def eci_calibration(self, calibration_dataset, k, alpha=0.1):
        dataloader = DataLoader(calibration_dataset, batch_size=self.args.BATCH_SIZE, shuffle=False)

        self.eval()
        scores_per_class = {i: [] for i in range(self.num_base_classes)}

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)

                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in scores_per_class:
                        score = 1.0 - probs[i, label].item()
                        scores_per_class[label].append(score)

        # Compute quantile thresholds
        self.eci_thresholds_k = {}
        for cls, scores in scores_per_class.items():
            if len(scores) == 0:
                self.eci_thresholds_k[cls] = 1.0  # fallback
            else:
                scores_np = np.array(scores)
                n_val = len(scores_np)
                q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
                self.eci_thresholds_k[cls] = float(np.quantile(scores_np, q_level))
        print(f"ECII thresholds for detector {k}: {self.eci_thresholds_k}")
        self.eci_thresholds.append(self.eci_thresholds_k)
    def collect_cp_entropy(self, val_dataset, k):
        self.prompt_manager.eval()
        for sample in val_dataset:
            input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
                self.cp_entropy_scores.append(entropy)

    def fit(self, train_dataset, k):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.prompt_manager.train()
        for param in self.shared_encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.prompt_manager.get_all_ood_parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            total_correct = 0
            total_id = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                is_ood = torch.tensor([y.item() not in self.base_class_global_indices for y in labels], device=self.device)
                is_id = ~is_ood

                if is_id.any():
                    loss_id = loss_fn(logits[is_id], labels[is_id])
                    preds = torch.argmax(logits[is_id], dim=1)
                    total_correct += (preds == labels[is_id]).sum().item()
                    total_id += is_id.sum().item()
                else:
                    loss_id = torch.tensor(0.0, device=self.device, requires_grad=True)

                if is_ood.any() and is_id.any():
                    entropy_id = entropy[is_id].mean()
                    entropy_ood = entropy[is_ood].mean()
                    loss_margin = torch.clamp(self.args.OOD_MARGIN + entropy_id - entropy_ood, min=0.0)
                else:
                    loss_margin = torch.tensor(0.0, device=self.device, requires_grad=True)

                loss = loss_id + self.args.LAMBDA_ENTROPY * loss_margin

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.prompt_manager.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            acc = total_correct / total_id if total_id > 0 else 0.0
            print(f"[PromptTuning][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] Loss: {avg_loss:.4f} | ID Acc: {acc:.4f}")
    
    def fit_sub_classifiers(self, train_dataset, threshold_tau):
        from torch.utils.data import DataLoader
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # self.shared_encoder.eval()  # Allow encoder to be trainable for gradient flow

        for k in range(self.k_detectors):
            print(f"\n[Training Sub-classifier + COOP Prompt #{k}]")
            optimizer = torch.optim.Adam(
                list(self.prompt_manager.coop_prompts[k].parameters()) +
                list(self.sub_classifiers_[k].parameters()),
                lr=learning_rate
            )

            for epoch in range(self.args.N_EPOCHS_SUBCLASSIFIER):
                total_loss = 0
                self.train()
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # OOD detector to mask ID/OOD using prompt_manager.forward_ood
                    with torch.no_grad():
                        logits_ood = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                        probs_ood = softmax(logits_ood, dim=1)
                        entropy = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)
                        is_id = entropy <= self.args.CP_OOD_THRESHOLD

                    # Remove no_grad: allow encoder + prompt to be updated for score function
                    coop_logits = self.prompt_manager.forward_coop(k=k, input_ids=input_ids, attention_mask=attention_mask)
                    logits_sub = coop_logits

                    # Compute logits_zs for KL loss, zs_classifier remains frozen but allow gradient for input
                    with torch.no_grad():
                        outputs_zs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
                        cls_token_zs = outputs_zs.last_hidden_state[:, 0]
                        logits_zs = self.zs_classifier_.classifier(cls_token_zs)

                    loss = 0.0
                    if is_id.any():
                        loss_id = cross_entropy(logits_sub[is_id], labels[is_id])
                        loss += loss_id
                    if (~is_id).any():
                        loss_kl = kl_divergence_loss_from_logits(logits_sub[~is_id], logits_zs[~is_id])
                        loss += self.args.KL_COEFF * loss_kl

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Total Loss: {total_loss/len(dataloader):.4f}")

    def predict_proba(self, tokenized_sample):
        self.eval()
        input_ids = tokenized_sample["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = tokenized_sample["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            detector_scores = []
            probs_k = []
            for k in range(self.k_detectors):
                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                detector_scores.append(entropy.item())
                probs_k.append(probs.squeeze(0))

            max_k = int(np.argmax(detector_scores))
            selected_probs = probs_k[max_k]
            pred_class = int(torch.argmax(selected_probs))
            non_conf_score = 1.0 - selected_probs[pred_class].item()

            eci_thresh_dict = self.eci_thresholds[max_k]
            eci_thresh = eci_thresh_dict.get(pred_class, 1.0)

            if non_conf_score > eci_thresh:
                logits_zs = self.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits_zs, dim=1)
            else:
                coop_logits = self.prompt_manager.forward_coop(k=max_k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(coop_logits, dim=1)

        return probs.cpu().numpy().squeeze()



class DECOOPInferenceEngine:
    def __init__(self, model, eci_thresholds):
        self.model = model
        self.eci_thresholds = eci_thresholds  # List[ECIBasedOODDetector]

    def predict(self, tokenized_sample):
        input_ids = tokenized_sample["input_ids"].unsqueeze(0).to(self.model.device)
        attention_mask = tokenized_sample["attention_mask"].unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            detector_scores = []
            probs_k = []

            for k in range(self.model.k_detectors):
                logits = self.model.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                detector_scores.append(entropy.item())
                probs_k.append(probs.squeeze(0))

            # Step 1: select detector with highest entropy
            max_k = int(np.argmax(detector_scores))
            selected_probs = probs_k[max_k]

            # Step 2: get predicted class and its non-conformity score
            pred_class = int(torch.argmax(selected_probs))
            non_conf_score = 1.0 - selected_probs[pred_class].item()

            # Step 3: compare with ECII threshold
            eci_thresh_dict = self.eci_thresholds[max_k]
            eci_thresh = eci_thresh_dict.get(pred_class, 1.0)

            if non_conf_score > eci_thresh:
                # classified as OOD
                zs_logits = self.model.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(zs_logits, dim=1)
                return "OOD", probs.cpu().numpy().squeeze()
            else:
                # classified as ID
                coop_logits = self.model.prompt_manager.forward_coop(k=max_k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(coop_logits, dim=1)
                return "ID", probs.cpu().numpy().squeeze()

    def update_thresholds(self, tokenized_sample):
        pass