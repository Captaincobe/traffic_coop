import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoModel # Keep AutoTokenizer, AutoModelForMaskedLM for MLMZeroShotClassifier
from utils.utils import EarlyStopping, kl_divergence_loss_from_logits
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR


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
        self.classifier = nn.Linear(hidden_dim, num_classes) # , dtype=torch.float16

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
        # self.classifier = nn.Linear(self.embedding_dim, num_labels).to(torch.float32)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),                               
            nn.Dropout(0.2),                      
            nn.Linear(self.embedding_dim // 2, num_labels) 
        )
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
    
    def forward_coop_batch(self, input_ids, attention_mask):
        """
        批量计算所有 K 个 COOP Prompt 和分类器的 logits。
        input_ids:  (B, L)
        返回 logits: (B, C, K) - 其中 B 是批量大小，C 是类别数，K 是检测器数量
        """
        K = self.K 
        B, L = input_ids.shape
        
        # 堆叠所有 K 个 COOP Prompt 的嵌入，并将其扩展到批量大小
        prompt_bank = torch.stack([p.ln(p.prompt_embedding)
                                for p in self.coop_prompts])   # (K,Lp,H)
        prompt_bank = prompt_bank.unsqueeze(1).expand(-1, B, -1, -1)     # (K,B,Lp,H)
        prompt_bank = prompt_bank.reshape(B*K, self.prompt_length, -1)   # (B·K,Lp,H)

        # 编码器部分通常是冻结的，所以可以使用 no_grad
        with torch.no_grad():
            tok_embeds = self.encoder.embeddings(input_ids)              # (B,L,H)
        tok_embeds = tok_embeds.unsqueeze(0).expand(K, -1, -1, -1)       # (K,B,L,H)
        tok_embeds = tok_embeds.reshape(B*K, L, -1)                      # (B·K,L,H)

        # 拼接 Prompt 嵌入和 Token 嵌入
        full_embeds = torch.cat([prompt_bank, tok_embeds], 1)            # (B·K,L+Lp,H)

        # 构造完整的注意力掩码
        full_mask = torch.cat([
            torch.ones(B*K, self.prompt_length, dtype=attention_mask.dtype, device=self.device),
            attention_mask.unsqueeze(0).expand(K, -1, -1).reshape(B*K, L)
        ], 1)

        # 通过编码器获取输出
        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls = outputs.last_hidden_state[:,0]                             # (B·K,H)
        # 通过分类器获取 logits，并重新塑形为 (B, C, K)
        logits = self.classifier(cls).view(K, B, -1).permute(1,2,0)      # (B,C,K)
        return logits
    def forward_ood(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.ood_prompts[k], input_ids, attention_mask)

    def forward_coop(self, k, input_ids, attention_mask):
        return self._forward_with_prompt(self.coop_prompts[k], input_ids, attention_mask)

    # ---- return only parameters for the k‑th prompt ----
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
        # -------------------- Shared Encoder (LoRA) --------------------
        self.shared_encoder = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32,
            # low_cpu_mem_usage=True # 如果内存紧张可以打开
        ).to(self.device)

        for param in self.shared_encoder.parameters():
            param.requires_grad = False #

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
        self.base_class_global_indices = sorted(base_class_global_indices)
        self.base_global_set = set(self.base_class_global_indices)
        self.global2base = {g: i for i, g in enumerate(self.base_class_global_indices)}

    def eci_calibration(self, calibration_dataset, k, alpha=0.1):
        dataloader = DataLoader(calibration_dataset, batch_size=self.args.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

        self.eval()
        # scores_per_class = {i: [] for i in range(self.num_base_classes)}
        # 修正：scores_per_class 应该只为当前检测器实际会产生分数的局部索引进行初始化。
        # 这些局部索引是 self.global2base 的值域，也就是 0 到 len(self.base_global_set) - 1。
        scores_per_class = {local_idx: [] for local_idx in self.global2base.values()}

        # --- debug ---
        # print(f"\n[DEBUG ECII] Detector {k} calibration started.")
        # print(f"[DEBUG ECII] Detector {k}'s current self.base_global_set (ID global indices): {self.base_global_set}")
        # print(f"[DEBUG ECII] Detector {k}'s current self.global2base: {self.global2base}")
        # print(f"[DEBUG ECII] Expecting scores for Local Base Index 12 (Global Class 13). Is Global 13 in self.base_global_set? {13 in self.base_global_set}")
        # -----------------------------------------------------------

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                global_labels = batch["global_labels"].to(self.device)

                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(logits, dim=1)

                for i in range(len(global_labels)):
                    current_global_label = global_labels[i].item()
                    prob = probs[i].detach().cpu().numpy()
                    pred = np.argmax(prob)
                    score = 1.0 - prob[pred]  # non-conformity score: 1 - max prob

                    if current_global_label not in self.base_global_set:
                        continue  # 非基类直接跳过
                    
                    label_base = self.global2base[current_global_label]
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

    def fit(self, train_dataset, val_dataset, k):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_PROMPT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
        
        # Keep encoder frozen to save memory; only train prompt & classifier
        self.prompt_manager.train()

        optimizer = torch.optim.AdamW(
            self.prompt_manager.get_ood_parameters_k(k),
            lr=learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()
        num_training_steps = len(dataloader) * self.args.NUM_EPOCHS # 总步数

        if self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
            warmup_steps = int(num_training_steps * self.args.WARMUP_EPOCHS / self.args.NUM_EPOCHS)
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

            scheduler = LambdaLR(optimizer, lr_lambda)
        elif self.args.LR_SCHEDULER_TYPE == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.args.PLATEAU_FACTOR,
                patience=self.args.PLATEAU_PATIENCE,
                verbose=True
            )
            print(f"[Info] Using ReduceLROnPlateau for OOD prompt {k}.")

        early_stopping = EarlyStopping(patience=self.args.PLATEAU_PATIENCE, # 使用与 ReduceLROnPlateau 相同的patience
                                min_delta=0.0001, # 可以根据需要调整
                                verbose=True)
        scaler = GradScaler()

        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            total_correct = 0
            total_id = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with autocast():
                    logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                    probs = softmax(logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                # 判断是否属于基类：使用全局索引集合
                base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                is_id = torch.isin(labels, base_indices_tensor)
                is_ood = ~is_id

                if is_id.any():
                    # —— 全局标签 → 基类标签 —— 
                    mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                    for g, b in self.global2base.items():
                        mapping_tensor[g] = b
                    y_base = mapping_tensor[labels[is_id]]
                    loss_id = loss_fn(logits[is_id], y_base)
                    preds = torch.argmax(logits[is_id], dim=1)
                    total_correct += (preds == y_base).sum().item()
                    total_id += is_id.sum().item()
                else:
                    loss_id = torch.tensor(0.0, device=self.device, requires_grad=True)


                loss_margin = torch.tensor(0.0, device=self.device) # Ensure requires_grad=False by default with AMP
                if is_ood.any() and is_id.any():
                    entropy_id = entropy[is_id].mean()
                    entropy_ood = entropy[is_ood].mean()
                    loss_margin = torch.clamp(self.args.OOD_MARGIN + entropy_id - entropy_ood, min=0.0)
                elif is_ood.any(): # Only OOD samples in batch
                    entropy_ood = entropy[is_ood].mean()
                elif is_id.any(): # Only ID samples in batch
                    entropy_id = entropy[is_id].mean()


                loss = loss_id + self.args.LAMBDA_ENTROPY * loss_margin


                # optimizer.zero_grad()
                # loss.backward()
                # # torch.nn.utils.clip_grad_norm_(self.prompt_manager.parameters(), max_norm=1.0)
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(self.prompt_manager.get_ood_parameters_k(k), max_norm=1.0) # 梯度裁剪

                scaler.step(optimizer) 
                scaler.update()              

                # if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                #     scheduler.step() # CosineAnnealingLR是按步更新
                total_loss += loss.item()
                # 如果使用了CosineAnnealingLR，在这里步进调度器
                if self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                    # 注意：如果warmup_steps是基于总步数，这里需要调整
                    # 或者将scheduler的step放到epoch外，让它根据epoch总数来调度
                    # 为了早停，最好让调度器在epoch级别更新，如果它基于epoch的话
                    pass # CosineAnnealingLR 已经在外面根据总步数设置了

            avg_loss = total_loss / len(dataloader)
            acc = total_correct / total_id if total_id > 0 else 0.0
            print(f"[PromptTuning][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] | Loss: {avg_loss:.4f} | ID Acc: {acc:.4f}")
# Loss_id: {loss_id:4.f} | Loss_margin: {loss_margin:4.f} 


            # --- Validation ---
            self.eval() 
            total_val_loss = 0
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    val_input_ids = val_batch["input_ids"].to(self.device)
                    val_attention_mask = val_batch["attention_mask"].to(self.device)
                    val_global_labels = val_batch["global_labels"].to(self.device)

                    # 注意：验证集包含ID和OOD，但OOD detector只对ID样本进行分类
                    # 这里我们仍然使用 forward_ood，并计算ID样本的损失
                    val_logits = self.prompt_manager.forward_ood(k=k, input_ids=val_input_ids, attention_mask=val_attention_mask)

                    val_base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                    val_is_id = torch.isin(val_global_labels, val_base_indices_tensor)

                    if val_is_id.any():
                        # 将全局标签映射到本地基类索引
                        val_mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                        for g, b in self.global2base.items():
                            val_mapping_tensor[g] = b
                        val_y_base = val_mapping_tensor[val_global_labels[val_is_id]]
                        val_loss_id = loss_fn(val_logits[val_is_id], val_y_base)
                        total_val_loss += val_loss_id.item()
                    # 对于OOD样本，OOD prompt的loss_id部分不适用，但它们会影响熵，这里只关注ID分类性能
                    # 如果需要OOD相关的验证指标，需要单独计算（如OOD-AUROC）

            avg_val_loss = total_val_loss / (len(val_dataloader) if len(val_dataloader) > 0 else 1)
            print(f"[PromptTuning][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] Val Loss: {avg_val_loss:.4f}")

            # --- 学习率调度器更新 (ReduceLROnPlateau) ---
            if self.args.LR_SCHEDULER_TYPE == "plateau":
                scheduler.step(avg_val_loss) # 根据验证损失调整学习率

            # --- 早停判断 ---
            if early_stopping(avg_val_loss, self): # 传入当前模型实例
                print(f"Early stopping at epoch {epoch+1}!")
                break # 中断训练循环

        if early_stopping.early_stop and early_stopping.best_model_state:
            self.load_state_dict(early_stopping.best_model_state)
            print("Loaded best model state based on early stopping.")

        # ---- freeze the k‑th OOD prompt after finishing training ----
        for p in self.prompt_manager.ood_prompts[k].parameters():
            p.requires_grad = False

    
    def fit_sub_classifiers(self, train_dataset):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_SUBFIT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        scaler = GradScaler()


        for k in range(self.k_detectors):
            print(f"\n[Training Sub-classifier + COOP Prompt #{k}]")
            self.prompt_manager.train()
            optimizer = torch.optim.AdamW(
                # 训练COOP Prompt Manager的参数，Sub-classifier的参数，以及共享编码器的LoRA适配器
                list(self.prompt_manager.get_coop_parameters_k(k)) +
                list(self.sub_classifiers_[k].parameters()),
                lr=learning_rate
            )
            # === 学习率调度器设置 (针对 fit_sub_classifiers 方法) ===
            num_training_steps = len(dataloader) * self.args.N_EPOCHS_SUBCLASSIFIER

            if self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                warmup_steps = int(num_training_steps * self.args.WARMUP_EPOCHS / self.args.N_EPOCHS_SUBCLASSIFIER)
                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

                scheduler = LambdaLR(optimizer, lr_lambda)
            elif self.args.LR_SCHEDULER_TYPE == "plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.args.PLATEAU_FACTOR,
                    patience=self.args.PLATEAU_PATIENCE,
                    verbose=True
                )
                print(f"[Info] Using ReduceLROnPlateau for COOP prompt {k}.")
            else:
                scheduler = None
                print("[Info] No LR scheduler applied for COOP prompt training.")

            for epoch in range(self.args.N_EPOCHS_SUBCLASSIFIER):
                total_loss = 0
                self.train()
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    # OOD detector to mask ID/OOD using prompt_manager.forward_ood
                    with torch.no_grad():

                        logits_ood = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
                        probs_ood  = softmax(logits_ood, dim=1)
                        entropy    = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)

                        base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                        id_by_entropy = entropy <= self.args.CP_OOD_THRESHOLD
                        id_by_label   = torch.isin(labels, base_indices_tensor)
                        is_id = id_by_entropy & id_by_label            # 同时满足才算 ID
                    with autocast():
                        logits_sub = self.prompt_manager.forward_coop(k=k, input_ids=input_ids, attention_mask=attention_mask)

                        # Compute logits_zs for KL loss, zs_classifier remains frozen but allow gradient for input
                        with torch.no_grad():
                            outputs_zs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
                            cls_token_zs = outputs_zs.last_hidden_state[:, 0]
                            logits_zs = self.zs_classifier_.classifier(cls_token_zs)

                        loss = 0.0
                        if is_id.any():
                            mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                            for g, b in self.global2base.items():
                                mapping_tensor[g] = b
                            y_base = mapping_tensor[labels[is_id]]
                            loss_id = cross_entropy(logits_sub[is_id], y_base)
                            loss += loss_id
                        if (~is_id).any():
                            loss_kl = kl_divergence_loss_from_logits(logits_sub[~is_id], logits_zs[~is_id])
                            loss += self.args.KL_COEFF * loss_kl

                    # optimizer.zero_grad()
                    # loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    # optimizer.step()

                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    scaler.unscale_(optimizer) # 在裁剪前unscale
                    torch.nn.utils.clip_grad_norm_(
                        list(self.prompt_manager.get_coop_parameters_k(k)) +
                        list(self.sub_classifiers_[k].parameters()),
                        max_norm=1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                        scheduler.step() # CosineAnnealingLR是按步更新
                    total_loss += loss.item()
                print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Total Loss: {total_loss/len(dataloader):.4f}")
                if scheduler and self.args.LR_SCHEDULER_TYPE == "plateau":
                    scheduler.step(total_loss/len(dataloader)) # ReduceLROnPlateau是按epoch更新，传入监控指标

            # ---- freeze the k‑th COOP prompt after training ----
            for p in self.prompt_manager.coop_prompts[k].parameters():
                p.requires_grad = False


class DECOOPInferenceEngine:
    def __init__(self, model, eci_thresholds):
        self.model = model
        self.eci_thresholds = eci_thresholds  # List[ECIBasedOODDetector]

    def calibrate_q_hat(self, val_dataset, alpha_cp):
        """
        使用验证数据集校准全局共形预测 OOD 阈值 (q_hat)。
        计算出的 q_hat 将设置到 self.model.args.CP_OOD_THRESHOLD 中。
        """
        # 使用模型参数中的批量大小作为 DataLoader 的批量大小
        val_batch_size = self.model.args.BATCH_SIZE
        # 创建验证集的 DataLoader
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=16)

        all_non_conformity_scores_val = []

        # 获取模型的全局基类集合，用于过滤
        base_class_set = self.model.base_global_set


        if len(val_dataset) == 0:
            print("Validation set empty, fallback q_hat used.")
            q_hat = 0.9
        else:
            for batch_data in val_dataloader:
                # batch_prob_vectors 的形状是 (B, NUM_ALL_CLASSES)
                _, batch_prob_vectors = self.predict(batch_data)
                
                # 从批次数据中获取真实全局标签
                batch_global_labels = batch_data['global_labels'].cpu().numpy()

                # 逐个处理批次中的样本，计算并累积非一致性分数
                for b in range(batch_prob_vectors.shape[0]):
                    sample_global_label = batch_global_labels[b].item()
                    sample_probas = batch_prob_vectors[b] # (NUM_ALL_CLASSES,)

                    # 仅收集属于基类 (ID) 的样本分数
                    if sample_global_label not in base_class_set:
                        continue
                    
                    # 计算非一致性分数：1.0 - 模型预测该真实全局标签的概率
                    # sample_probas 已经是 NUM_ALL_CLASSES 维度，所以可以直接用全局标签作为索引
                    try:
                        score = 1.0 - sample_probas[sample_global_label]
                    except IndexError: 
                        print(f"Warning: IndexError during score calculation for global label {sample_global_label}. Probas shape: {sample_probas.shape}")
                        score = 1.0 # 错误回退值
                    
                    all_non_conformity_scores_val.append(score)

            all_non_conformity_scores_val = np.array(all_non_conformity_scores_val)
            n_val = len(all_non_conformity_scores_val)
            q_level = np.ceil((n_val + 1) * (1 - alpha_cp)) / n_val
            q_hat = np.quantile(all_non_conformity_scores_val, q_level) if n_val > 0 else 0.9
        print(all_non_conformity_scores_val)
        # 将计算出的 q_hat 设置到模型的参数中
        self.model.args.CP_OOD_THRESHOLD = q_hat
        print(f"Validation complete. q_hat = {q_hat:.4f}")

    def predict(self, batch):
        self.model.eval() # 设置模型为评估模式
        
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        raw_texts = batch.get("raw_text", None)

        B = input_ids.size(0) # 获取批次大小

        # 修正：prob_vectors 初始化为总类别数 (self.model.num_all_classes) 的维度，并用 0 填充
        prob_vectors = torch.zeros(B, self.model.num_all_classes, device=self.model.device) # <--- 修正这一行

        pred_types = [] 

        with torch.no_grad():
            # 1) 批量获取所有 K 个 OOD 检测器的 logits

            logits_ood_batch = self.model.prompt_manager.forward_ood_batch(
                input_ids=input_ids, attention_mask=attention_mask
            ) # (B, C, K)
            probs_ood_batch = softmax(logits_ood_batch, dim=1) # (B, C, K)

            max_probs, pred_bases = probs_ood_batch.max(dim=1) # max_probs: (B, K), pred_bases: (B, K)
            non_conf = 1.0 - max_probs # 非一致性分数 (B, K)

            best_k_indices = torch.argmin(non_conf, dim=1) # (B,) tensor

            is_new_mask = torch.zeros(B, dtype=torch.bool, device=self.model.device)

            for b in range(B):
                sample_best_k = best_k_indices[b].item()
                sample_pred_base = pred_bases[b, sample_best_k].item()
                sample_best_non_conf = float(non_conf[b, sample_best_k])

                sample_ecii_thresh = self.eci_thresholds[sample_best_k].get(sample_pred_base, 1.0)
                sample_cp_thresh = getattr(self.model.args, "CP_OOD_THRESHOLD", 1.0)

                is_new_sample = (sample_best_non_conf > sample_ecii_thresh) or (sample_best_non_conf > sample_cp_thresh)
                is_new_mask[b] = torch.tensor(is_new_sample, dtype=torch.bool, device=self.model.device) # <--- 修正这一行

                pred_types.append("NEW" if is_new_sample else "ID")
            
            # --- 批量处理 "NEW" 样本 ---
            if is_new_mask.any():
                new_sample_indices = torch.where(is_new_mask)[0]
                
                if raw_texts is None: 
                    print("Warning: raw_texts not available in batch for MLMZeroShotClassifier. Using empty strings.")
                    texts_for_mlm = [""] * len(new_sample_indices)
                else:
                    texts_for_mlm = [raw_texts[idx] for idx in new_sample_indices.cpu().tolist()]
                
                zs_probs_batch, _ = self.model.zs_mlm_classifier.predict(texts_for_mlm) # (Num_new_samples, C_all)
                # 直接赋值，因为 zs_probs_batch 已经是 C_all 维度
                prob_vectors[new_sample_indices] = zs_probs_batch.to(self.model.device) 

            # --- 批量处理 "ID" 样本 ---
            if (~is_new_mask).any():
                id_sample_indices = torch.where(~is_new_mask)[0]
                
                logits_coop_all_k = self.model.prompt_manager.forward_coop_batch(
                    input_ids=input_ids[id_sample_indices],
                    attention_mask=attention_mask[id_sample_indices]
                ) # (Num_id_samples, C_base, K)

                best_k_for_id_samples = best_k_indices[id_sample_indices]
                
                best_k_expanded = best_k_for_id_samples.view(-1, 1, 1).expand(-1, logits_coop_all_k.size(1), -1)
                
                selected_coop_logits = torch.gather(logits_coop_all_k, dim=2, index=best_k_expanded).squeeze(2) # (Num_id_samples, C_base)
                

            probs_id_batch = softmax(selected_coop_logits, dim=1) # (Num_id_samples, C_base)
            global_indices_of_base_classes = torch.tensor(
                sorted(list(self.model.base_global_set)), # 获取模型最终设置的全局基类索引
                dtype=torch.long, device=self.model.device
            )
            
            # 使用高级索引将 C_base 概率赋值到 C_all 维度的正确位置
            # 例如：prob_vectors[行索引, 列索引] = 值
            # id_sample_indices[:, None] 扩展为 (Num_id_samples, 1) 用于行索引
            # global_indices_of_base_classes 包含了列索引
            prob_vectors[id_sample_indices[:, None], global_indices_of_base_classes] = probs_id_batch.to(self.model.device) 

        return pred_types, prob_vectors.cpu().numpy() # 返回预测类型列表和 NumPy 数组

    def predict_batch(self, dataset):
        """
        对给定数据集进行批量推理，并收集所有预测结果。
        该方法将负责 DataLoader 的创建和批次循环。

        Returns:
            tuple: (point_preds_global, prob_matrix_all_classes, predictions_conformal_sets)
        """
        print(f"\nPredicting on Test set with Conformal Prediction...")
        
        test_batch_size = self.model.args.BATCH_SIZE 
        test_dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=16)

        all_point_preds_global = []
        all_prob_matrix_all_classes = []
        all_predictions_conformal_sets = []
        
        # 使用模型最终设置的整体基类集合，用于映射和共形集构建
        # base_class_list_sorted = sorted(list(self.model.base_global_set))

        for batch_idx, batch_data in enumerate(test_dataloader):
            # 调用自身的 predict 方法进行批量预测
            batch_pred_types, batch_prob_vectors = self.predict(batch_data) # batch_prob_vectors 的形状是 (B, NUM_ALL_CLASSES)

            # 逐个处理批次中的每个样本结果
            for b in range(batch_prob_vectors.shape[0]):
                sample_pred_type = batch_pred_types[b] # 当前样本的预测类型 ("ID" 或 "NEW")
                sample_probas = batch_prob_vectors[b] # 当前样本的概率向量 (NUM_ALL_CLASSES,)

                # 存储点预测结果 (直接是全局索引)
                pred_global_idx = np.argmax(sample_probas)
                all_point_preds_global.append(pred_global_idx)

                all_prob_matrix_all_classes.append(sample_probas)

                cp_set = []
                current_q_hat = self.model.args.CP_OOD_THRESHOLD 
                for global_idx_in_all_classes in range(self.model.args.NUM_ALL_CLASSES): 
                    if 1.0 - sample_probas[global_idx_in_all_classes] <= current_q_hat:
                        cp_set.append(global_idx_in_all_classes)
                all_predictions_conformal_sets.append(cp_set)

        point_preds_global = np.array(all_point_preds_global)
        prob_matrix_all_classes = np.array(all_prob_matrix_all_classes)
        predictions_conformal_sets = all_predictions_conformal_sets
        
        return point_preds_global, prob_matrix_all_classes, predictions_conformal_sets

    