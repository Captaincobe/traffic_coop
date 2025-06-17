import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoModel
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

        with torch.no_grad(): # MLMZeroShotClassifier 保持冻结，仅用于推理
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, cls_token=None):
        if cls_token is not None:
            return self.classifier(cls_token)
        if inputs_embeds is not None:
            output = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        return self.classifier(cls_token)

# New MLP for numerical feature extraction
class NumericalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(NumericalFeatureExtractor, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SharedRoBERTaPromptLearner(nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embedding = nn.Parameter(torch.empty(prompt_length, embedding_dim, dtype=torch.float32))
        nn.init.normal_(self.prompt_embedding, std=0.02)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        prompt_embed = self.ln(self.prompt_embedding).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_embed, input_embeds], dim=1)


class PromptLearnerManager(nn.Module):
    # Now takes fused_embedding_dim as input for classifier
    def __init__(self, shared_encoder, num_labels, prompt_length, K=4, device="cuda", fused_embedding_dim=None):
        super().__init__()
        self.device = device
        self.K = K
        self.prompt_length = prompt_length
        self.encoder = shared_encoder # 这里的 self.encoder 就是 LLMTrafficDECOOP.shared_encoder
        self.embedding_dim = self.encoder.config.hidden_size # LLM's output dim

        if fused_embedding_dim is None:
            raise ValueError("fused_embedding_dim must be provided for PromptLearnerManager.")

        self.ood_prompts = nn.ModuleList([
            SharedRoBERTaPromptLearner(prompt_length, self.embedding_dim) for _ in range(K)
        ])
        self.coop_prompts = nn.ModuleList([
            SharedRoBERTaPromptLearner(prompt_length, self.embedding_dim) for _ in range(K)
        ])
        # Classifier now takes fused_embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_embedding_dim, fused_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fused_embedding_dim // 2, num_labels)
        )

    # _forward_with_prompt now accepts features as well
    def _forward_with_prompt(self, prompt_module, input_ids, attention_mask, features):
        batch_size = input_ids.size(0)

        # Textual Path
        with torch.no_grad(): # Embeddings are frozen
            input_embeds = self.encoder.embeddings(input_ids)
        full_embeds = prompt_module(input_embeds) # [B, L + prompt_length, H]
        prompt_mask = torch.ones((batch_size, self.prompt_length), dtype=attention_mask.dtype).to(attention_mask.device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls_token_text = outputs.last_hidden_state[:, 0] # Textual embedding (B, H)

        # Numerical Path: features are passed directly from LLMTrafficDECOOP
        # Numerical feature extractor is external to PromptLearnerManager
        # We expect features to be already processed into an embedding by LLMTrafficDECOOP
        numerical_embedding = features # Assume 'features' here is already numerical embedding

        # Fusion: Concatenate textual and numerical embeddings
        fused_embedding = torch.cat([cls_token_text, numerical_embedding], dim=1) # (B, H + NumFE_OutputDim)

        logits = self.classifier(fused_embedding)
        return logits

    # forward_ood_batch now accepts features as well
    def forward_ood_batch(self, input_ids, attention_mask, features):
        K = self.K
        B, L = input_ids.shape

        prompt_bank = torch.stack([p.ln(p.prompt_embedding)
                                for p in self.ood_prompts])   # (K,Lp,H)
        prompt_bank = prompt_bank.unsqueeze(1).expand(-1, B, -1, -1)     # (K,B,Lp,H)
        prompt_bank = prompt_bank.reshape(B*K, self.prompt_length, -1)   # (B·K,Lp,H)

        with torch.no_grad(): # Apply no_grad to embeddings layer
            tok_embeds = self.encoder.embeddings(input_ids)              # (B,L,H)
        tok_embeds = tok_embeds.unsqueeze(0).expand(K, -1, -1, -1)       # (K,B,L,H)
        tok_embeds = tok_embeds.reshape(B*K, L, -1)                      # (B·K,L,H)

        full_embeds = torch.cat([prompt_bank, tok_embeds], 1)            # (B·K,L+Lp,H)

        full_mask = torch.cat([
            torch.ones(B*K, self.prompt_length, dtype=attention_mask.dtype, device=self.device),
            attention_mask.unsqueeze(0).expand(K, -1, -1).reshape(B*K, L)
        ], 1)

        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls_text_embeddings = outputs.last_hidden_state[:,0] # (B*K, H)

        # Expand numerical features to match B*K for concatenation
        numerical_embedding_expanded = features.unsqueeze(0).expand(K, -1, -1).reshape(B*K, features.size(-1))

        # Fusion: Concatenate textual and numerical embeddings
        fused_embedding = torch.cat([cls_text_embeddings, numerical_embedding_expanded], dim=1)

        logits = self.classifier(fused_embedding).view(K, B, -1).permute(1,2,0)      # (B,C,K)
        return logits

    # forward_coop_batch now accepts features as well
    def forward_coop_batch(self, input_ids, attention_mask, features):
        K = self.K
        B, L = input_ids.shape

        prompt_bank = torch.stack([p.ln(p.prompt_embedding)
                                for p in self.coop_prompts])   # (K,Lp,H)
        prompt_bank = prompt_bank.unsqueeze(1).expand(-1, B, -1, -1)     # (K,B,Lp,H)
        prompt_bank = prompt_bank.reshape(B*K, self.prompt_length, -1)   # (B·K,Lp,H)

        with torch.no_grad(): # Apply no_grad to embeddings layer
            tok_embeds = self.encoder.embeddings(input_ids)              # (B,L,H)
        tok_embeds = tok_embeds.unsqueeze(0).expand(K, -1, -1, -1)       # (K,B,L,H)
        tok_embeds = tok_embeds.reshape(B*K, L, -1)                      # (B·K,L,H)

        full_embeds = torch.cat([prompt_bank, tok_embeds], 1)            # (B·K,L+Lp,H)

        full_mask = torch.cat([
            torch.ones(B*K, self.prompt_length, dtype=attention_mask.dtype, device=self.device),
            attention_mask.unsqueeze(0).expand(K, -1, -1).reshape(B*K, L)
        ], 1)

        outputs = self.encoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        cls_text_embeddings = outputs.last_hidden_state[:,0]                             # (B·K,H)

        numerical_embedding_expanded = features.unsqueeze(0).expand(K, -1, -1).reshape(B*K, features.size(-1))

        fused_embedding = torch.cat([cls_text_embeddings, numerical_embedding_expanded], dim=1)

        logits = self.classifier(fused_embedding).view(K, B, -1).permute(1,2,0)      # (B,C,K)
        return logits

    def forward_ood(self, k, input_ids, attention_mask, features): # Add features
        return self._forward_with_prompt(self.ood_prompts[k], input_ids, attention_mask, features)

    def forward_coop(self, k, input_ids, attention_mask, features): # Add features
        return self._forward_with_prompt(self.coop_prompts[k], input_ids, attention_mask, features)

    def get_ood_parameters_k(self, k: int):
        trainable_params = list(self.ood_prompts[k].parameters()) + list(self.classifier.parameters())
        # If shared_encoder's parameters are trainable, include them
        for param in self.encoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_coop_parameters_k(self, k: int):
        trainable_params = list(self.coop_prompts[k].parameters()) + list(self.classifier.parameters())
        # If shared_encoder's parameters are trainable, include them
        for param in self.encoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params


class LLMTrafficDECOOP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.llm_model_name = args.LLM_MODEL_NAME
        self.device = args.DEVICE
        self.k_detectors = args.K_DETECTORS
        self.PROMPT_LENGTH = args.PROMPT_LENGTH

        self.input_dim = args.INPUT_DIM # Numerical input dim
        self.numerical_fe_output_dim = args.NUM_FE_OUTPUT_DIM # Output dim of numerical feature extractor

        # -------------------- Shared Encoder (BERT/RoBERTa) --------------------
        self.shared_encoder = AutoModel.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float32, # Explicitly use float32 for BERT model
            # low_cpu_mem_usage=True # 如果内存紧张可以打开
        ).to(self.device)

        self.hidden_dim = self.shared_encoder.config.hidden_size # LLM's output dim
        self.fused_embedding_dim = self.hidden_dim + self.numerical_fe_output_dim # Fusion dim


        # ====== 微调 BERT/RoBERTa 最后一层的修改 =======
        # 默认冻结所有参数
        for param in self.shared_encoder.parameters():
            param.requires_grad = False

        # 解冻 BERT 编码器的最后 N 层
        N_layers_to_unfreeze = getattr(args, 'N_BERT_LAYERS_TO_UNFREEZE', 1)
        print(f"[Info] Unfreezing last {N_layers_to_unfreeze} layers of {self.llm_model_name} encoder.")

        for i in range(self.shared_encoder.config.num_hidden_layers - N_layers_to_unfreeze,
                       self.shared_encoder.config.num_hidden_layers):
            for param in self.shared_encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        # 额外解冻 LayerNorm 和/或 Pooler (BERT specific)
        if hasattr(self.shared_encoder.encoder, 'AfterLayerNorm'): # RoBERTa specific
             for param in self.shared_encoder.encoder.AfterLayerNorm.parameters():
                  param.requires_grad = True
        if hasattr(self.shared_encoder, 'pooler') and self.shared_encoder.pooler is not None: # BERT specific
            for param in self.shared_encoder.pooler.parameters():
                param.requires_grad = True

        # 如果需要，也可以解冻 embeddings 层（会增加更多参数和计算量）
        # for param in self.shared_encoder.embeddings.parameters():
        #     param.requires_grad = True
        # ===============================================

        # -------------------- Numerical Feature Extractor (New MLP) --------------------
        self.numerical_feature_extractor = NumericalFeatureExtractor(
            input_dim=self.input_dim,
            hidden_dims=args.NUM_FE_HIDDEN_DIMS,
            output_dim=self.numerical_fe_output_dim
        ).to(self.device)
        # Ensure numerical_feature_extractor weights are also float32
        self.numerical_feature_extractor.float()


        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES
        self.base_class_set = set(range(self.num_base_classes))

        # -------------------- Classifiers --------------------
        ## Zero-shot classifier (frozen)
        self.zs_mlm_classifier = MLMZeroShotClassifier(
            model_name=args.LLM_MODEL_NAME,
            label_token_map=args.LABEL_TOKEN_MAP,
            device=self.device
        )

        ## Prompt Learner Manager (OOD + COOP prompts)
        # Pass fused_embedding_dim to PromptLearnerManager
        self.prompt_manager = PromptLearnerManager(
            shared_encoder=self.shared_encoder,
            num_labels=self.num_base_classes,
            prompt_length=self.PROMPT_LENGTH,
            K=self.k_detectors,
            device=self.device,
            fused_embedding_dim=self.fused_embedding_dim # Pass fused dimension
        ).to(self.device)
        # Ensure prompt_manager (classifier part) is also float32
        self.prompt_manager.classifier.float()

        self.eci_thresholds = []


    def set_base_class_global_indices(self, base_class_global_indices):
        self.base_class_global_indices = sorted(base_class_global_indices)
        self.base_global_set = set(self.base_class_global_indices)
        self.global2base = {g: i for i, g in enumerate(self.base_class_global_indices)}

    def eci_calibration(self, calibration_dataset, k, alpha=0.1):
        dataloader = DataLoader(calibration_dataset, batch_size=self.args.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

        self.eval()
        scores_per_class = {local_idx: [] for local_idx in self.global2base.values()}

        print(f"\n[ECII] Detector {k} calibration started.")
        print(f"[ECII] Detector {k}'s current self.base_global_set (ID global indices): {self.base_global_set}")
        print(f"[ECII] Detector {k}'s current global2base mapping: {self.global2base}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device) # Get numerical features
                global_labels = batch["global_labels"].to(self.device)

                # Process numerical features
                # Explicitly cast features to match model's default dtype (float32)
                numerical_embedding = self.numerical_feature_extractor(features.to(torch.float32))

                # Pass numerical embedding to prompt_manager
                logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask, features=numerical_embedding)
                probs = softmax(logits, dim=1)

                for i in range(len(global_labels)):
                    current_global_label = global_labels[i].item()
                    prob = probs[i].detach().cpu().numpy()
                    pred = np.argmax(prob)
                    score = 1.0 - prob[pred]  # non-conformity score: 1 - max prob

                    if current_global_label not in self.base_global_set:
                        continue

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

        self.prompt_manager.train()
        self.numerical_feature_extractor.train() # Set numerical feature extractor to train mode

        # Collect all trainable parameters for the optimizer
        params_to_optimize = list(self.prompt_manager.get_ood_parameters_k(k)) + list(self.numerical_feature_extractor.parameters())

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            weight_decay=self.args.WEIGHT_DECAY,
            lr=learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()
        num_training_steps = len(dataloader) * self.args.NUM_EPOCHS

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
        else:
            scheduler = None

        early_stopping = EarlyStopping(patience=self.args.PLATEAU_PATIENCE,
                                min_delta=0.0001,
                                verbose=True)
        scaler = GradScaler()
        from collections import Counter
        val_global_labels_all = [s['global_labels'].item() for s in val_dataset]
        print(f"\nUnique Global Labels in Val Dataset: {sorted(list(set(val_global_labels_all)))}")
        print(f"Global Label Counts in Val Dataset: {Counter(val_global_labels_all)}")
        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            total_correct = 0
            total_id = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device) # Get numerical features
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                with autocast():
                    # Explicitly cast features to match model's default dtype (float32)
                    numerical_embedding = self.numerical_feature_extractor(features.to(torch.float32))
                    logits = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask, features=numerical_embedding)
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

                entropy_id_mean = torch.tensor(0.0, device=self.device, requires_grad=True)
                entropy_ood_mean = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_margin = torch.tensor(0.0, device=self.device, requires_grad=True) 
                if is_ood.any() and is_id.any():
                    entropy_id_mean = entropy[is_id].mean()
                    entropy_ood_mean = entropy[is_ood].mean()
                    loss_margin = torch.clamp(self.args.OOD_MARGIN + entropy_id_mean - entropy_ood_mean, min=0.0)
                elif is_ood.any():
                    entropy_ood_mean = entropy[is_ood].mean()
                elif is_id.any():
                    entropy_id_mean = entropy[is_id].mean()

                # epsilon_tensor = torch.tensor(1e-9, device=self.device, requires_grad=True)
                # loss = loss + epsilon_tensor
                loss = loss_id + self.args.LAMBDA_ENTROPY * loss_margin

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # 裁剪 prompt_manager 和 numerical_feature_extractor 的所有可训练参数
                all_trainable_params = list(self.prompt_manager.get_ood_parameters_k(k)) + list(self.numerical_feature_extractor.parameters())
                torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                    scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            acc = total_correct / total_id if total_id > 0 else 0.0
            print(f"[PromptTuning][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] | Loss: {avg_loss:.4f} | ID Acc: {acc:.4f} | E_ID: {entropy_id_mean:.4f} | E_OOD: {entropy_ood_mean:.4f}")

            if epoch >= 60 and (epoch - 60) % 2 == 0: #
                self.eval()
                self.numerical_feature_extractor.eval() # Set to eval mode for validation
                total_val_loss = 0
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                        val_input_ids = val_batch["input_ids"].to(self.device)
                        val_attention_mask = val_batch["attention_mask"].to(self.device)
                        val_features = val_batch["features"].to(self.device) # Get numerical features
                        val_global_labels = val_batch["global_labels"].to(self.device)

                        # Explicitly cast features to match model's default dtype (float32)
                        val_numerical_embedding = self.numerical_feature_extractor(val_features.to(torch.float32))
                        val_logits = self.prompt_manager.forward_ood(k=k, input_ids=val_input_ids, attention_mask=val_attention_mask, features=val_numerical_embedding)

                        val_base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                        val_is_id = torch.isin(val_global_labels, val_base_indices_tensor)

                        if val_is_id.any():
                            val_mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                            for g, b in self.global2base.items():
                                val_mapping_tensor[g] = b
                            val_y_base = val_mapping_tensor[val_global_labels[val_is_id]]
                            val_loss_id = loss_fn(val_logits[val_is_id], val_y_base)
                            total_val_loss += val_loss_id.item()

                    avg_val_loss = total_val_loss / (len(val_dataloader) if len(val_dataloader) > 0 else 1)
                    print(f"[PromptTuning][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] Val Loss: {avg_val_loss:.4f}")

                if self.args.LR_SCHEDULER_TYPE == "plateau":
                    scheduler.step(avg_val_loss)

                if early_stopping(avg_val_loss, self):
                    print(f"Early stopping at epoch {epoch+1}!")
                    break

        if early_stopping.early_stop and early_stopping.best_model_state:
            self.load_state_dict(early_stopping.best_model_state)
            print("Loaded best model state based on early stopping.")

        # ---- Freeze all trainable parameters from this phase ----
        for p in self.prompt_manager.ood_prompts[k].parameters():
            p.requires_grad = False
        for p in self.numerical_feature_extractor.parameters():
            p.requires_grad = False

        # Re-enable specific layers of shared_encoder for the next phase if needed
        # (This logic is handled by the overall LLMTrafficDECOOP init and fit_sub_classifiers)


    def fit_sub_classifiers(self, train_dataset, val_dataset):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_SUBFIT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

        scaler = GradScaler()


        for k in range(self.k_detectors):
            print(f"\n[Training Sub-classifier + COOP Prompt #{k}]")
            self.prompt_manager.train()
            self.numerical_feature_extractor.train() # Set numerical feature extractor to train mode

            params_to_optimize = list(self.prompt_manager.get_coop_parameters_k(k)) + list(self.numerical_feature_extractor.parameters())

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                weight_decay=self.args.WEIGHT_DECAY,
                lr=learning_rate
            )

            if self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                warmup_steps = int(self.args.N_EPOCHS_SUBCLASSIFIER * self.args.WARMUP_EPOCHS)
                num_training_epochs = self.args.N_EPOCHS_SUBCLASSIFIER
                def lr_lambda(current_epoch: int):
                    if current_epoch < warmup_steps:
                        return float(current_epoch) / float(max(1, warmup_steps))
                    progress = float(current_epoch - warmup_steps) / float(max(1, num_training_epochs - warmup_steps))
                    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

                scheduler = LambdaLR(optimizer, lr_lambda)
                print(f"[Info] Using CosineAnnealingLR with warmup for COOP prompt {k}.")
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

            early_stopping = EarlyStopping(patience=self.args.PLATEAU_PATIENCE,
                                         min_delta=0.0001,
                                         verbose=True)

            # for ood_det in self.ood_classifiers:
            #     ood_det.eval()
            # zs_mlm_classifier 保持冻结

            for epoch in range(self.args.N_EPOCHS_SUBCLASSIFIER):
                self.train() # This trains the COOP classifier, not the entire DECOOP model (which might have frozen parts)
                self.prompt_manager.train() # Ensure prompt_manager is in train mode
                self.numerical_feature_extractor.train() # Ensure numerical feature extractor is in train mode

                total_loss = 0
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    features = batch["features"].to(self.device) # Get numerical features
                    labels = batch["labels"].to(self.device)
                    raw_texts = batch["raw_text"]

                    optimizer.zero_grad()
                    with torch.no_grad():
                        # Explicitly cast features to match model's default dtype (float32)
                        numerical_embedding = self.numerical_feature_extractor(features.to(torch.float32))
                        logits_ood = self.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask, features=numerical_embedding)
                        probs_ood  = softmax(logits_ood, dim=1)
                        entropy    = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)

                        base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                        id_by_entropy = entropy <= self.args.CP_OOD_THRESHOLD
                        id_by_label   = torch.isin(labels, base_indices_tensor)
                        is_id = id_by_entropy & id_by_label

                    with autocast():
                        # numerical_embedding is already computed above
                        logits_coop = self.prompt_manager.forward_coop(k=k, input_ids=input_ids, attention_mask=attention_mask, features=numerical_embedding)

                        loss = 0.0
                        if is_id.any():
                            mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                            for g, b in self.global2base.items():
                                mapping_tensor[g] = b
                            y_base = mapping_tensor[labels[is_id]]
                            loss_id = cross_entropy(logits_coop[is_id], y_base)
                            loss += loss_id
                        if (~is_id).any():
                            with torch.no_grad():
                                outputs_zs_probs, _ = self.zs_mlm_classifier.predict([raw_texts[i] for i in torch.where(~is_id)[0]])
                                outputs_zs_probs = outputs_zs_probs.to(self.device)
                                zs_logits_for_kl_distillation = torch.log(outputs_zs_probs + 1e-8)
                                zs_logits_for_kl_distillation = zs_logits_for_kl_distillation[:, self.base_class_global_indices]

                            loss_kl = kl_divergence_loss_from_logits(logits_coop[~is_id], zs_logits_for_kl_distillation)
                            loss += self.args.KL_COEFF * loss_kl

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    all_trainable_params = list(self.prompt_manager.get_coop_parameters_k(k)) + list(self.numerical_feature_extractor.parameters())
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(dataloader)
                print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Train Loss: {avg_train_loss:.4f}")

                if epoch >= 20 and (epoch - 20) % 2 == 0:
                    self.eval() # Set overall model to eval
                    self.prompt_manager.eval() # Set prompt manager to eval
                    self.numerical_feature_extractor.eval() # Set numerical feature extractor to eval
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_input_ids = val_batch["input_ids"].to(self.device)
                            val_attention_mask = val_batch["attention_mask"].to(self.device)
                            val_features = val_batch["features"].to(self.device) # Get numerical features
                            val_global_labels = val_batch["global_labels"].to(self.device)
                            val_raw_texts = val_batch["raw_text"]

                            # Explicitly cast features to match model's default dtype (float32)
                            val_numerical_embedding = self.numerical_feature_extractor(val_features.to(torch.float32))
                            val_logits_ood = self.prompt_manager.forward_ood(k=k, input_ids=val_input_ids, attention_mask=val_attention_mask, features=val_numerical_embedding)
                            val_probs_ood  = softmax(val_logits_ood, dim=1)
                            val_entropy    = -torch.sum(val_probs_ood * torch.log(val_probs_ood + 1e-8), dim=1)

                            val_base_indices_tensor = torch.tensor(self.base_class_global_indices, device=self.device)
                            val_id_by_entropy = val_entropy <= self.args.CP_OOD_THRESHOLD
                            val_id_by_label   = torch.isin(val_global_labels, val_base_indices_tensor)
                            val_is_id = val_id_by_entropy & val_id_by_label

                            val_logits_coop = self.prompt_manager.forward_coop(k=k, input_ids=val_input_ids, attention_mask=val_attention_mask, features=val_numerical_embedding)

                            val_batch_loss = 0.0
                            if val_is_id.any():
                                val_mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                                for g, b in self.global2base.items():
                                    val_mapping_tensor[g] = b
                                val_y_base = val_mapping_tensor[val_global_labels[val_is_id]]
                                val_loss_id = cross_entropy(val_logits_coop[val_is_id], val_y_base)
                                val_batch_loss += val_loss_id
                            if (~val_is_id).any():
                                val_zs_probs_ood_samples, _ = self.zs_mlm_classifier.predict([val_raw_texts[i] for i in torch.where(~val_is_id)[0]])
                                val_zs_probs_ood_samples = val_zs_probs_ood_samples.to(self.device)
                                val_zs_logits_ood_samples = torch.log(val_zs_probs_ood_samples + 1e-8)
                                val_zs_logits_for_kl_distillation = val_zs_logits_ood_samples[:, self.base_class_global_indices]

                                val_loss_kl = kl_divergence_loss_from_logits(val_logits_coop[~val_is_id], val_zs_logits_for_kl_distillation)
                                val_batch_loss += self.args.KL_COEFF * val_loss_kl

                            total_val_loss += val_batch_loss.item()

                    avg_val_loss = total_val_loss / (len(val_dataloader) if len(val_dataloader) > 0 else 1)
                    print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Val Loss: {avg_val_loss:.4f}")


                    if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                        scheduler.step()
                    elif scheduler and self.args.LR_SCHEDULER_TYPE == "plateau":
                        scheduler.step(avg_val_loss)

                    if early_stopping(avg_val_loss, self):
                        print(f"Early stopping at epoch {epoch+1} for Sub-classifier {k}!")
                        break

            if early_stopping.early_stop and early_stopping.best_model_state:
                self.load_state_dict(early_stopping.best_model_state)
                print(f"Loaded best model state for Sub-classifier {k} based on early stopping.")
            else:
                 print(f"Training of Sub-classifier {k} finished without early stopping.")


            # ---- Freeze all trainable parameters from this phase ----
            for p in self.prompt_manager.get_coop_parameters_k(k):
                p.requires_grad = False
            for p in self.numerical_feature_extractor.parameters():
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
        val_batch_size = self.model.args.BATCH_SIZE
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=16)

        all_non_conformity_scores_val = []

        base_class_set = self.model.base_global_set


        if len(val_dataset) == 0:
            print("Validation set empty, fallback q_hat used.")
            q_hat = 0.9
        else:
            for batch_data in val_dataloader:
                # batch_prob_vectors 的形状是 (B, NUM_ALL_CLASSES)
                _, batch_prob_vectors = self.predict(batch_data)

                batch_global_labels = batch_data['global_labels'].cpu().numpy()

                # 逐个处理批次中的样本，计算并累积非一致性分数
                for b in range(batch_prob_vectors.shape[0]):
                    sample_global_label = batch_global_labels[b].item()
                    sample_probas = batch_prob_vectors[b] # (NUM_ALL_CLASSES,)

                    # 仅收集属于基类 (ID) 的样本分数
                    if sample_global_label not in base_class_set:
                        continue

                    # 计算非一致性分数：1.0 - 模型预测该真实全局标签的概率
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
        self.model.eval()

        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        features = batch["features"].to(self.model.device) # Get numerical features
        raw_texts = batch.get("raw_text", None)

        B = input_ids.size(0)

        prob_vectors = torch.zeros(B, self.model.num_all_classes, device=self.model.device)
        pred_types=[]
        with torch.no_grad():
            # Process numerical features
            # Explicitly cast features to match model's default dtype (float32)
            numerical_embedding = self.model.numerical_feature_extractor(features.to(torch.float32))

            logits_ood_batch = self.model.prompt_manager.forward_ood_batch(
                input_ids=input_ids, attention_mask=attention_mask, features=numerical_embedding
            ) # (B, C, K)
            probs_ood_batch = softmax(logits_ood_batch, dim=1)

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
                is_new_mask[b] = torch.tensor(is_new_sample, dtype=torch.bool, device=self.model.device)

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

                # Process numerical features for ID samples
                # Explicitly cast features to match model's default dtype (float32)
                numerical_embedding_id_samples = self.model.numerical_feature_extractor(features[id_sample_indices].to(torch.float32))

                logits_coop_all_k = self.model.prompt_manager.forward_coop_batch(
                    input_ids=input_ids[id_sample_indices],
                    attention_mask=attention_mask[id_sample_indices],
                    features=numerical_embedding_id_samples
                ) # (Num_id_samples, C_base, K)

                best_k_for_id_samples = best_k_indices[id_sample_indices]

                best_k_expanded = best_k_for_id_samples.view(-1, 1, 1).expand(-1, logits_coop_all_k.size(1), -1)

                selected_coop_logits = torch.gather(logits_coop_all_k, dim=2, index=best_k_expanded).squeeze(2)


            probs_id_batch = softmax(selected_coop_logits, dim=1) # (Num_id_samples, C_base)
            global_indices_of_base_classes = torch.tensor(
                sorted(list(self.model.base_global_set)),
                dtype=torch.long, device=self.model.device
            )

            # 使用高级索引将 C_base 概率赋值到 C_all 维度的正确位置
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

        for batch_idx, batch_data in enumerate(test_dataloader):
            batch_pred_types, batch_prob_vectors = self.predict(batch_data)

            for b in range(batch_prob_vectors.shape[0]):
                sample_pred_type = batch_pred_types[b]
                sample_probas = batch_prob_vectors[b]

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