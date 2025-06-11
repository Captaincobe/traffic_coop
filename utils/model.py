import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoModel
from utils.utils import kl_divergence_loss_from_logits
from torch.utils.data import DataLoader

class MLMZeroShotClassifier:
    def __init__(self, model_name, label_token_map, template="This traffic is [MASK].", device="cuda"):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.template = template
        self.label_token_map = label_token_map

        # ------------------------------------------------------------------
        # Convert label ➜ token‑id  ——  Scheme B: automatically add new
        # whole‑word tokens to the tokenizer when the default BPE split
        # would produce multiple sub‑tokens.  This keeps every verbalizer
        # one‑to‑one with a single token so that masked‑LM logits are
        # comparable across labels.
        # ------------------------------------------------------------------
        self.label_token_ids = {}

        # 1) Gather all words that need to be appended to the vocab
        tokens_to_add = []
        for word in label_token_map.values():
            if len(self.tokenizer.tokenize(word)) != 1:
                tokens_to_add.append(word)

        # 2) Add them *once* and resize the model’s embedding matrix
        if tokens_to_add:
            n_added = self.tokenizer.add_tokens(tokens_to_add, special_tokens=False)
            if n_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"[Info] Added {n_added} new tokenizer tokens for MLM labels: {tokens_to_add}")

        # 3) Build the final label→token‑id mapping (now guaranteed 1‑to‑1)
        for label, word in label_token_map.items():
            tokens = self.tokenizer.tokenize(word)
            if len(tokens) != 1:
                # Defensive fallback; should not happen after the addition step
                print(f"[Warning] Label '{label}' still maps to multiple tokens: {tokens}")
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
            self.label_token_ids[label] = token_id

    def predict(self, input_texts):
        """
        Parameters
        ----------
        input_texts : Union[str, List[str]]
            Raw flow‑text descriptions. A single string will be automatically converted
            to a one‑element list.

        Returns
        -------
        probs : torch.Tensor, shape = (B, C)
            Soft‑max probabilities over C labels for each sample in the batch.
        label_names : List[str]
            The label order corresponding to the second dimension of probs.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Assemble the prompt: "<flow text> This traffic is [MASK]."
        prompts = [
            f"{t} This traffic is {self.tokenizer.mask_token}."
            for t in input_texts
        ]

        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Locate the [MASK] position for every sample
        mask_positions = (batch.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        with torch.no_grad():
            logits = self.model(**batch).logits        # (B, L, V)

        # Select logits corresponding to the [MASK] token
        mask_logits = logits[mask_positions]          # (B, V)

        # Keep only the logits of label verbalizer tokens
        label_token_ids = list(self.label_token_ids.values())
        scores = mask_logits[:, label_token_ids]      # (B, C)

        probs = torch.softmax(scores, dim=-1)         # (B, C)

        return probs.cpu(), list(self.label_token_ids.keys())


class SharedEncoderClassifier(nn.Module):
    def __init__(self, shared_encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = shared_encoder
        self.classifier = nn.Linear(hidden_dim, num_classes, dtype=torch.float32)

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
        # nn.init.kaiming_normal_(self.prompt_embedding,nonlinearity='leaky_relu') # xavier_normal_
        nn.init.normal_(self.prompt_embedding, std=0.02)
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
        full_embeds = prompt_module(input_embeds) # [B, L + prompt_length, H]
        prompt_mask = torch.ones((batch_size, self.prompt_length), dtype=attention_mask.dtype).to(attention_mask.device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        # logits = torch.clamp(logits, min=-10, max=10)
        return logits
        # return self.classifier(cls_token)

    def forward_ood_batch(self, input_ids, attention_mask):
        """
        input_ids:  (B, L)
        返回 logits: (B, C, K)
        """
        K = self.K 
        B, L = input_ids.shape
        # (K, L, H) 软提示，先 stack；再 expand 到 batch
        prompt_bank = torch.stack([p.ln(p.prompt_embedding)
                                for p in self.ood_prompts])   # (K,Lp,H)
        prompt_bank = prompt_bank.unsqueeze(1).expand(-1, B, -1, -1)     # (K,B,Lp,H)
        prompt_bank = prompt_bank.reshape(B*K, self.prompt_length, -1)   # (B·K,Lp,H)

        # word embeddings (no_grad)
        with torch.no_grad():
            tok_embeds = self.encoder.embeddings(input_ids)              # (B,L,H)
        tok_embeds = tok_embeds.unsqueeze(0).expand(K, -1, -1, -1)       # (K,B,L,H)
        tok_embeds = tok_embeds.reshape(B*K, L, -1)                      # (B·K,L,H)

        full_embeds = torch.cat([prompt_bank, tok_embeds], 1)            # (B·K,L+Lp,H)

        full_mask = torch.cat([
            torch.ones(B*K, self.prompt_length, dtype=attention_mask.dtype, device=self.device),
            attention_mask.unsqueeze(0).expand(K, -1, -1).reshape(B*K, L)
        ], 1)

        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls = outputs.last_hidden_state[:,0]                             # (B·K,H)
        logits = self.classifier(cls).view(K, B, -1).permute(1,2,0)      # (B,C,K)
        return logits
    def forward_ood(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.ood_prompts[k], input_ids, attention_mask)

    def forward_coop(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.coop_prompts[k], input_ids, attention_mask)

    # def get_all_ood_parameters(self):
    #     return list(self.ood_prompts.parameters()) + list(self.classifier.parameters())

    # def get_all_coop_parameters(self):
    #     return list(self.coop_prompts.parameters()) + list(self.classifier.parameters())

    # ---- New helper: return only parameters for the k‑th prompt ----
    def get_ood_parameters_k(self, k: int):
        return list(self.ood_prompts[k].parameters()) + list(self.classifier.parameters())

    def get_coop_parameters_k(self, k: int):
        return list(self.coop_prompts[k].parameters()) + list(self.classifier.parameters())
    

class LLMTrafficDECOOP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.llm_model_name = args.LLM_MODEL_NAME
        self.device = args.DEVICE
        self.k_detectors = args.K_DETECTORS
        self.PROMPT_LENGTH = args.PROMPT_LENGTH
        # -------------------- Shared Encoder (frozen) --------------------
        self.shared_encoder = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32,
            # low_cpu_mem_usage=True
        ).to(self.device)

        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        # Embeddings remain frozen for efficiency
        self.hidden_dim = self.shared_encoder.config.hidden_size

        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES
        # 相对索引集合仅在少数地方备用；主判断用 base_global_set
        self.base_class_set = set(range(self.num_base_classes))

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
            prompt_length=self.PROMPT_LENGTH,
            K=self.k_detectors,
            device=self.device
        ).to(self.device)

        # MLM Zero-shot classifier for fallback
        self.zs_mlm_classifier = MLMZeroShotClassifier(
            model_name=args.LLM_MODEL_NAME,
            label_token_map=args.LABEL_TOKEN_MAP,
            device=self.device
        )

        # self.cp_entropy_scores = []
        self.eci_thresholds = []


    def set_base_class_global_indices(self, base_class_global_indices):
        # 全局索引（如 1,2,3,4,5,7,9,10,12,13）→ 排序
        self.base_class_global_indices = sorted(base_class_global_indices)
        # 快速判别集合
        self.base_global_set = set(self.base_class_global_indices)
        # 建立全局 idx → 基类相对 idx（0…NUM_BASE_CLASSES-1）
        self.global2base = {g: i for i, g in enumerate(self.base_class_global_indices)}

    def eci_calibration(self, calibration_dataset, k, alpha=0.1):
        dataloader = DataLoader(calibration_dataset, batch_size=self.args.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

        self.eval()
        scores_per_class = {i: [] for i in range(self.num_base_classes)}

        # Collect per-class non-conformity scores with debug output
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                # 正确获取全局标签！
                global_labels = batch["global_labels"].to(self.device) # <--- 修正：使用 batch["global_labels"]

                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)

                # 遍历批次中的每个样本
                for i in range(len(global_labels)): # <--- 修正：遍历 global_labels
                    current_global_label = global_labels[i].item() # <--- 获取当前样本的正确全局标签
                    prob = probs[i].detach().cpu().numpy()
                    pred = np.argmax(prob)
                    score = 1.0 - prob[pred]  # non-conformity score: 1 - max prob

                    # 正确的判断：如果当前全局标签不在当前检测器的 base_global_set 中，则跳过
                    if current_global_label not in self.base_global_set: # <--- 修正：用正确的全局标签进行判断
                        continue  # 非基类直接跳过
                    
                    # 正确的映射：将全局标签映射到局部基类索引
                    label_base = self.global2base[current_global_label] # <--- 修正：用正确的全局标签进行映射
                    scores_per_class[label_base].append(score)

        # Compute quantile thresholds per class, with warnings and debug info
        self.eci_thresholds_k = {}
        print(f"[ECII] Per-class non-conformity score stats for detector {k}:")
        for cls, scores in scores_per_class.items():
            if len(scores) == 0:
                self.eci_thresholds_k[cls] = 1.0
                print(f"[ECII] Warning: no validation samples for class {cls}, using fallback threshold=1.0")
            else:
                scores_np = np.array(scores)
                n_val = len(scores_np)
                q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
                min_score, max_score = scores_np.min(), scores_np.max()
                if np.allclose(min_score, max_score, atol=1e-3) and max_score > 0.95:
                    print(f"[ECII] Warning: class {cls} score distribution is very narrow (min={min_score:.4f}, max={max_score:.4f})")
                try:
                    thresh = float(np.quantile(scores_np, q_level))
                except Exception:
                    thresh = 1.0
                self.eci_thresholds_k[cls] = thresh
                print(f"[ECII] class {cls} | q@{1-alpha:.2f} = {self.eci_thresholds_k[cls]:.4f}, N={len(scores)}")
        print(f"[ECII] thresholds for detector {k}: {self.eci_thresholds_k}")
        self.eci_thresholds.append(self.eci_thresholds_k)

    def fit(self, train_dataset, k):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

        # Keep encoder frozen to save memory; only train prompt & classifier
        self.prompt_manager.train()

        optimizer = torch.optim.Adam(
            self.prompt_manager.get_ood_parameters_k(k),
            lr=learning_rate
        )
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

                # 判断是否属于基类：使用全局索引集合
                is_ood = torch.tensor([y.item() not in self.base_global_set for y in labels], device=self.device)
                is_id = ~is_ood

                if is_id.any():
                    # —— 全局标签 → 基类标签 —— 
                    y_base = torch.tensor(
                        [self.global2base[y.item()] for y in labels[is_id]],
                        device=self.device
                    )
                    loss_id = loss_fn(logits[is_id], y_base)
                    preds = torch.argmax(logits[is_id], dim=1)
                    total_correct += (preds == y_base).sum().item()
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

        # ---- freeze the k‑th OOD prompt after finishing training ----
        for p in self.prompt_manager.ood_prompts[k].parameters():
            p.requires_grad = False
    
    def fit_sub_classifiers(self, train_dataset):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)


        for k in range(self.k_detectors):
            print(f"\n[Training Sub-classifier + COOP Prompt #{k}]")
            optimizer = torch.optim.Adam(
                self.prompt_manager.get_coop_parameters_k(k) +
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
                        probs_ood  = softmax(logits_ood, dim=1)
                        entropy    = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)

                        id_by_entropy = entropy <= self.args.CP_OOD_THRESHOLD
                        id_by_label   = torch.tensor([y.item() in self.base_global_set for y in labels],
                                                    device=self.device)
                        is_id = id_by_entropy & id_by_label            # 同时满足才算 ID

                    logits_sub = self.prompt_manager.forward_coop(k=k, input_ids=input_ids, attention_mask=attention_mask)

                    # Compute logits_zs for KL loss, zs_classifier remains frozen but allow gradient for input
                    with torch.no_grad():
                        outputs_zs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
                        cls_token_zs = outputs_zs.last_hidden_state[:, 0]
                        logits_zs = self.zs_classifier_.classifier(cls_token_zs)

                    loss = 0.0
                    if is_id.any():
                        y_base = torch.tensor(
                            [self.global2base[y.item()] for y in labels[is_id]],
                            device=self.device
                        )
                        loss_id = cross_entropy(logits_sub[is_id], y_base)
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

            # ---- freeze the k‑th COOP prompt after training ----
            for p in self.prompt_manager.coop_prompts[k].parameters():
                p.requires_grad = False


class DECOOPInferenceEngine:
    def __init__(self, model, eci_thresholds):
        self.model = model
        self.eci_thresholds = eci_thresholds  # List[ECIBasedOODDetector]

    def predict(self, tokenized_sample):
        self.model.eval()
        input_ids      = tokenized_sample["input_ids"].unsqueeze(0).to(self.model.device)
        attention_mask = tokenized_sample["attention_mask"].unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            # 1)  (1,C,K) logits
            logits_batch = self.model.prompt_manager.forward_ood_batch(
                input_ids=input_ids, attention_mask=attention_mask
            )                                # (1,C,K)
            probs_batch  = softmax(logits_batch, dim=1).squeeze(0)   # (C,K)

            max_probs, pred_bases = probs_batch.max(dim=0)           # (K,), (K,)
            non_conf = 1.0 - max_probs                               # (K,)
            best_k   = int(torch.argmin(non_conf).item())
            best_pred_base = int(pred_bases[best_k].item())
            best_non_conf  = float(non_conf[best_k])

            # 2) ecii
            ecii_thresh = self.eci_thresholds[best_k].get(best_pred_base, 1.0)
            cp_thresh   = getattr(self.model.args, "CP_OOD_THRESHOLD", 1.0)
            is_new      = (best_non_conf > ecii_thresh) or (best_non_conf > cp_thresh)

            if is_new:
                pred_type = "NEW"
                print("NEW")
                zs_probs, _ = self.model.zs_mlm_classifier.predict(
                    tokenized_sample.get("raw_text","")
                )
                prob_vec = zs_probs[0]               # (C_base,)
            else:
                pred_type = "ID"
                coop_logits = self.model.prompt_manager.forward_coop(
                    k=best_k, input_ids=input_ids, attention_mask=attention_mask
                )
                prob_vec = softmax(coop_logits, dim=1).cpu().numpy().squeeze()

        return pred_type, prob_vec

    