import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoTokenizer, AutoModel
from utils.loss import calculate_batch_entropy_from_logits, kl_divergence_loss_from_logits

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
        logits = torch.clamp(logits, min=-10, max=10)
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

        # 初始化共享 RoBERTa encoder
        self.shared_encoder = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        self.hidden_dim = self.shared_encoder.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES

        # ZS classifier
        self.zs_classifier_ = SharedEncoderClassifier(self.shared_encoder, self.hidden_dim, self.num_base_classes).to(self.device)

        # Detector classifiers（K 个）
        self.new_class_detectors_ = nn.ModuleList([
            SharedEncoderClassifier(self.shared_encoder, self.hidden_dim, self.num_base_classes).to(self.device)
            for _ in range(self.k_detectors)
        ])

        # Sub-classifiers（K 个）
        self.sub_classifiers_ = nn.ModuleList([
            SharedEncoderClassifier(self.shared_encoder, self.hidden_dim, self.num_base_classes).to(self.device)
            for _ in range(self.k_detectors)
        ])

        # Prompt learners
        self.prompt_manager = PromptLearnerManager(
            shared_encoder=self.shared_encoder,
            num_labels=self.num_base_classes,
            prompt_length=PROMPT_LENGTH,
            K=self.k_detectors,
            device=self.device
        ).to(self.device)

    def set_base_class_global_indices(self, base_class_global_indices):
        self.base_class_global_indices = base_class_global_indices

    def fit(self, train_dataset, k):
        from torch.utils.data import DataLoader
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        self.pseudo_ood_classes = getattr(self, "pseudo_ood_classes", {})
        label_set = set()
        for sample in train_dataset:
            label_set.add(sample['labels'].item())
        self.pseudo_ood_classes[k] = label_set

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # print(f"\n[Training PromptLearner OOD #{k}]")
        optimizer = torch.optim.Adam(self.prompt_manager.get_all_ood_parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        def model_forward(input_ids, attention_mask):
            return self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)

        self.prompt_manager.train()
        for epoch in range(self.args.N_EPOCHS_DETECTOR):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = model_forward(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                is_ood = torch.tensor([y.item() in self.pseudo_ood_classes[k] for y in labels], device=self.device)
                is_id = ~is_ood

                if is_id.any():
                    loss_id = cross_entropy(logits[is_id], labels[is_id])
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
                torch.nn.utils.clip_grad_norm_(self.prompt_manager.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
            print(f"[PromptTuning][Epoch {epoch+1}/{self.args.N_EPOCHS_DETECTOR}] Loss: {total_loss/len(dataloader):.4f}")
    
    def fit_sub_classifiers(self, train_dataset, threshold_tau):
        from torch.utils.data import DataLoader
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.shared_encoder.eval()

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

                    with torch.no_grad():
                        logits_ood = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                        probs_ood = softmax(logits_ood, dim=1)
                        entropy = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)
                        is_id = entropy <= threshold_tau

                    with torch.no_grad():
                        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
                        cls_token = outputs.last_hidden_state[:, 0]
                    logits_sub = self.sub_classifiers_[k].classifier(cls_token)

                    with torch.no_grad():
                        logits_zs = self.zs_classifier_.classifier(cls_token)

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
            outputs = self.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask).to(torch.float32)
            probs = softmax(outputs, dim=1)
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
            for k in range(self.model.k_detectors):
                logits = self.model.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                detector_scores.append(entropy.item())

            max_score = max(detector_scores)
            max_k = detector_scores.index(max_score)
            eci = self.eci_thresholds[max_k]

            if max_score > self.model.args.CP_OOD_THRESHOLD:
                zs_logits = self.model.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(zs_logits, dim=1)
                return "OOD", probs.cpu().numpy().squeeze()
            else:
                coop_logits = self.model.prompt_manager.forward_coop(k=max_k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(coop_logits, dim=1)
                return "ID", probs.cpu().numpy().squeeze()

    def update_thresholds(self, tokenized_sample):
        pass