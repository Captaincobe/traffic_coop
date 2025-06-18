import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.functional import softmax, cross_entropy, kl_div, log_softmax
from utils.utils import EarlyStopping
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

# Import MLMZeroShotClassifier from the LLM utils model
# from utils.model import MLMZeroShotClassifier # 新增导入
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

    def eval(self):
        """
        Make the wrapped masked‑LM enter evaluation mode and
        return self so the call can be chained just like with
        nn.Module.eval().
        """
        self.model.eval()
        return self



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

# --- Helper for KL Div ---
def kl_divergence_loss_from_logits(pred_logits, target_logits_detached):
    pred_log_probs = log_softmax(pred_logits, dim=1)
    target_probs_detached = softmax(target_logits_detached, dim=1).detach()
    return kl_div(pred_log_probs, target_probs_detached, reduction='batchmean', log_target=False)

# --- Linear Layer ---
class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# --- MLPClassifier ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, use_bn=True):
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError("`hidden_dims` must be a non-empty list or tuple for this MLP architecture.")
        nhid = hidden_dims[0]

        dropout = 0.5
        super(MLPClassifier, self).__init__()

        self.Linear1 = Linear(input_dim, nhid * 2, dropout, bias=True)
        self.Linear2 = Linear(nhid * 2, nhid, dropout, bias=True)
        self.Linear3 = Linear(nhid, num_classes, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(nhid * 2)
            self.bn3 = nn.BatchNorm1d(nhid)

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        if self.use_bn: # 确保在 Linear3 前应用 BN
            x = self.bn3(x)
        x = F.relu(self.Linear3(x)) # Note: F.relu on final output is unusual for classification logits. If this is meant to be logits, remove relu.
        return x


class LLMTrafficDECOOP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.DEVICE
        self.k_detectors = args.K_DETECTORS

        self.input_dim = args.INPUT_DIM
        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES

        # --- Zero‑shot classifier (MLM) ---
        self.zs_classifier_ = MLMZeroShotClassifier(
            model_name=args.PRE_MODEL,          # e.g. "bert-base-uncased"
            label_token_map=args.LABEL_TOKEN_MAP,    # map global label → verbalizer
            device=self.device
        )
        self.zs_classifier_.eval()  # frozen; inference only

        self.ood_classifiers = nn.ModuleList([
            MLPClassifier(
                input_dim=self.input_dim,
                hidden_dims=args.MLP_HIDDEN_DIMS_OOD,
                num_classes=self.num_base_classes
            ).to(self.device)
            for _ in range(self.k_detectors)
        ])

        self.coop_classifiers = nn.ModuleList([
            MLPClassifier(
                input_dim=self.input_dim,
                hidden_dims=args.MLP_HIDDEN_DIMS_COOP,
                num_classes=self.num_base_classes
            ).to(self.device)
            for _ in range(self.k_detectors)
        ])

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
        print(f"[ECII] Detector {k}'s current base_global_set (ID global indices): {self.base_global_set}")
        print(f"[ECII] Detector {k}'s current global2base mapping: {self.global2base}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                features = batch["features"].to(self.device)
                global_labels = batch["global_labels"].to(self.device)

                logits = self.ood_classifiers[k](features)
                probs = softmax(logits, dim=1)

                for i in range(len(global_labels)):
                    current_global_label = global_labels[i].item()

                    if current_global_label not in self.base_global_set:
                        continue

                    label_base = self.global2base[current_global_label]

                    prob_true_class = probs[i, label_base].item()
                    score = 1.0 - prob_true_class

                    scores_per_class[label_base].append(score)

        self.eci_thresholds_k = {}
        print(f"[ECII] Per-class non-conformity score stats for detector {k}:")
        for cls_local_idx, scores in scores_per_class.items():
            if len(scores) == 0:
                self.eci_thresholds_k[cls_local_idx] = 1.0
                print(f"[ECII] Warning: no validation samples for local class {cls_local_idx}, using fallback threshold=1.0")
            else:
                scores_np = np.array(scores)
                n_val = len(scores_np)
                q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val

                if n_val > 0 and np.allclose(scores_np.min(), scores_np.max(), atol=1e-6) and scores_np.max() > 0.95:
                    print(f"[ECII] Warning: local class {cls_local_idx} score distribution is very narrow (min={scores_np.min():.4f}, max={scores_np.max():.4f})")

                try:
                    thresh = float(np.quantile(scores_np, q_level))
                except Exception as e:
                    print(f"[ECII] Error calculating quantile for local class {cls_local_idx}: {e}. Using fallback threshold=1.0")
                    thresh = 1.0

                self.eci_thresholds_k[cls_local_idx] = thresh
                print(f"[ECII] class {cls_local_idx} | q@{1-alpha:.2f} = {self.eci_thresholds_k[cls_local_idx]:.4f}, N={len(scores)}")
        print(f"[ECII] Thresholds for detector {k}: {self.eci_thresholds_k}")
        self.eci_thresholds.append(self.eci_thresholds_k)

    # --- New methods for batching MLP forward passes ---
    def forward_ood_batch_mlp(self, features):
        all_logits = []
        for k in range(self.k_detectors):
            self.ood_classifiers[k].eval()
            logits_k = self.ood_classifiers[k](features)
            all_logits.append(logits_k.unsqueeze(-1))
        return torch.cat(all_logits, dim=-1)

    def forward_coop_batch_mlp(self, features):
        all_logits = []
        for k in range(self.k_detectors):
            self.coop_classifiers[k].eval()
            logits_k = self.coop_classifiers[k](features)
            all_logits.append(logits_k.unsqueeze(-1))
        return torch.cat(all_logits, dim=-1)


    def fit(self, train_dataset, val_dataset, k):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_RATE_FIT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

        self.ood_classifiers[k].train()

        optimizer = torch.optim.AdamW(
            self.ood_classifiers[k].parameters(),
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
            print(f"[Info] Using ReduceLROnPlateau for OOD detector {k}.")
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
            total_id_samples = 0

            self.ood_classifiers[k].train()

            for batch_idx, batch in enumerate(dataloader):
                features = batch["features"].to(self.device)
                global_labels = batch["global_labels"].to(self.device)

                optimizer.zero_grad()
                with autocast():
                    logits = self.ood_classifiers[k](features)
                    probs = softmax(logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                is_id = torch.tensor([label.item() in self.base_global_set for label in global_labels], dtype=torch.bool, device=self.device)
                is_ood = ~is_id

                loss_id = torch.tensor(0.0, device=self.device, requires_grad=True)
                if is_id.any():
                    mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                    for g, b in self.global2base.items():
                        mapping_tensor[g] = b
                    y_base = mapping_tensor[global_labels[is_id]]

                    if (y_base == -1).any():
                        raise ValueError("Invalid local label (-1) found for ID samples during OOD detector training. Check data mapping.")

                    loss_id = loss_fn(logits[is_id], y_base)

                    preds = torch.argmax(logits[is_id], dim=1)
                    total_correct += (preds == y_base).sum().item()
                    total_id_samples += is_id.sum().item()

                loss_margin = torch.tensor(0.0, device=self.device, requires_grad=False)
                entropy_id_mean = torch.tensor(0.0, device=self.device)
                entropy_ood_mean = torch.tensor(0.0, device=self.device)

                if is_ood.any() and is_id.any():
                    entropy_id_mean = entropy[is_id].mean()
                    entropy_ood_mean = entropy[is_ood].mean()
                    loss_margin = torch.clamp(self.args.OOD_MARGIN + entropy_id_mean - entropy_ood_mean, min=0.0)
                elif is_ood.any():
                    entropy_ood_mean = entropy[is_ood].mean()
                elif is_id.any():
                    entropy_id_mean = entropy[is_id].mean()

                loss = loss_id + self.args.LAMBDA_ENTROPY * loss_margin

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.ood_classifiers[k].parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                    scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            acc = total_correct / total_id_samples if total_id_samples > 0 else 0.0
            print(f"[DetectorTraining][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] | Loss: {avg_loss:.4f} | ID Acc: {acc:.4f} | E_ID: {entropy_id_mean:.4f} | E_OOD: {entropy_ood_mean:.4f}")

            if epoch >= 60 and (epoch - 60) % 2 == 0:
                self.eval()
                self.ood_classifiers[k].eval()
                total_val_loss = 0
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_dataloader):
                        val_features = val_batch["features"].to(self.device)
                        val_global_labels = val_batch["global_labels"].to(self.device)

                        val_logits = self.ood_classifiers[k](val_features)

                        val_is_id = torch.tensor([label.item() in self.base_global_set for label in val_global_labels], dtype=torch.bool, device=self.device)

                        if val_is_id.any():
                            val_mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                            for g, b in self.global2base.items():
                                val_mapping_tensor[g] = b
                            val_y_base = val_mapping_tensor[val_global_labels[val_is_id]]
                            val_loss_id = loss_fn(val_logits[val_is_id], val_y_base)
                            total_val_loss += val_loss_id.item()

                    avg_val_loss = total_val_loss / (len(val_dataloader) if len(val_dataloader) > 0 else 1)
                    print(f"[DetectorTraining][Epoch {epoch+1}/{self.args.NUM_EPOCHS}] Val Loss: {avg_val_loss:.4f}")

                if scheduler and self.args.LR_SCHEDULER_TYPE == "plateau":
                    scheduler.step(avg_val_loss)

                if early_stopping(avg_val_loss, self):
                    print(f"Early stopping at epoch {epoch+1} for OOD detector {k}!")
                    break

        if early_stopping.early_stop and early_stopping.best_model_state:
            self.load_state_dict(early_stopping.best_model_state)
            print("Loaded best model state based on early stopping.")

        for p in self.ood_classifiers[k].parameters():
            p.requires_grad = False

    def fit_sub_classifiers(self, train_dataset, val_dataset):
        batch_size = self.args.BATCH_SIZE
        learning_rate = self.args.LEARNING_SUBFIT

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

        scaler = GradScaler()

        for k in range(self.k_detectors):
            print(f"\n[Training COOP Classifier #{k}]")
            self.coop_classifiers[k].train()

            optimizer = torch.optim.AdamW(
                self.coop_classifiers[k].parameters(),
                weight_decay=self.args.WEIGHT_DECAY,
                lr=learning_rate
            )

            if self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                warmup_epochs = int(self.args.N_EPOCHS_SUBCLASSIFIER * self.args.WARMUP_EPOCHS)
                num_training_epochs = self.args.N_EPOCHS_SUBCLASSIFIER
                def lr_lambda(current_epoch: int):
                    if current_epoch < warmup_epochs:
                        return float(current_epoch) / float(max(1, warmup_epochs))
                    progress = float(current_epoch - warmup_epochs) / float(max(1, num_training_epochs - warmup_epochs))
                    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
                scheduler = LambdaLR(optimizer, lr_lambda)
                print(f"[Info] Using CosineAnnealingLR with warmup for COOP classifier {k}.")
            elif self.args.LR_SCHEDULER_TYPE == "plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.args.PLATEAU_FACTOR,
                    patience=self.args.PLATEAU_PATIENCE,
                    verbose=True
                )
                print(f"[Info] Using ReduceLROnPlateau for COOP classifier {k}.")
            else:
                scheduler = None
                print("[Info] No LR scheduler applied for COOP classifier training.")

            early_stopping = EarlyStopping(patience=self.args.PLATEAU_PATIENCE,
                                         min_delta=0.0001,
                                         verbose=True)

            for ood_det in self.ood_classifiers:
                ood_det.eval()
            self.zs_classifier_.eval()


            for epoch in range(self.args.N_EPOCHS_SUBCLASSIFIER):
                self.train()
                self.coop_classifiers[k].train()
                total_loss = 0
                for batch in dataloader:
                    features = batch["features"].to(self.device) # Fused features
                    global_labels = batch["global_labels"].to(self.device)

                    optimizer.zero_grad()

                    with torch.no_grad():
                        logits_ood = self.ood_classifiers[k](features)
                        probs_ood  = softmax(logits_ood, dim=1)
                        entropy    = -torch.sum(probs_ood * torch.log(probs_ood + 1e-8), dim=1)

                        is_id = torch.tensor([label.item() in self.base_global_set for label in global_labels], dtype=torch.bool, device=self.device)
                        is_ood = ~is_id

                    with autocast():
                        logits_coop = self.coop_classifiers[k](features)

                        loss = 0.0
                        if is_id.any():
                            mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                            for g, b in self.global2base.items():
                                mapping_tensor[g] = b
                            y_base = mapping_tensor[global_labels[is_id]]

                            if (y_base == -1).any():
                                raise ValueError("Invalid local label (-1) found for ID samples during COOP classifier training. Check data mapping.")

                            loss_id = cross_entropy(logits_coop[is_id], y_base)
                            loss += loss_id

                        if is_ood.any():
                            with torch.no_grad():
                                # Now, zs_classifier_ is an MLP, it takes features directly
                                zs_logits_ood_samples = self.zs_classifier_(features[is_ood])
                                zs_logits_for_kl_distillation = zs_logits_ood_samples[:, self.base_class_global_indices]

                            loss_kl = kl_divergence_loss_from_logits(
                                logits_coop[is_ood],
                                zs_logits_for_kl_distillation
                            )
                            loss += self.args.KL_COEFF * loss_kl

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.coop_classifiers[k].parameters(),
                        max_norm=1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(dataloader)
                print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Train Loss: {avg_train_loss:.4f}")

                if epoch >= 20 and (epoch - 20) % 2 == 0:
                    self.eval()
                    self.coop_classifiers[k].eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_features = val_batch["features"].to(self.device) # Fused features
                            val_global_labels = val_batch["global_labels"].to(self.device)

                            val_logits_ood = self.ood_classifiers[k](val_features)
                            val_probs_ood  = softmax(val_logits_ood, dim=1)
                            val_entropy    = -torch.sum(val_probs_ood * torch.log(val_probs_ood + 1e-8), dim=1)

                            val_is_id = torch.tensor([label.item() in self.base_global_set for label in val_global_labels], dtype=torch.bool, device=self.device)
                            val_is_ood = ~val_is_id

                            val_logits_coop = self.coop_classifiers[k](val_features)

                            val_batch_loss = 0.0
                            if val_is_id.any():
                                val_mapping_tensor = torch.full((self.num_all_classes,), -1, device=self.device, dtype=torch.long)
                                for g, b in self.global2base.items():
                                    val_mapping_tensor[g] = b
                                val_y_base = val_mapping_tensor[val_global_labels[val_is_id]]
                                val_loss_id = cross_entropy(val_logits_coop[val_is_id], val_y_base)
                                val_batch_loss += val_loss_id
                            # if val_is_ood.any():
                            #     # Now, zs_classifier_ is an MLP, it takes features directly
                            #     val_zs_logits_ood_samples = self.zs_classifier_(val_features[val_is_ood])
                            #     val_zs_logits_for_kl_distillation = val_zs_logits_ood_samples[:, self.base_class_global_indices]

                            #     val_loss_kl = kl_divergence_loss_from_logits(val_logits_coop[val_is_ood], val_zs_logits_for_kl_distillation)
                            #     val_batch_loss += self.args.KL_COEFF * val_loss_kl

                            total_val_loss += val_batch_loss.item()

                    avg_val_loss = total_val_loss / (len(val_dataloader) if len(val_dataloader) > 0 else 1)
                    print(f"[Epoch {epoch+1}/{self.args.N_EPOCHS_SUBCLASSIFIER}] Val Loss: {avg_val_loss:.4f}")

                    if scheduler and self.args.LR_SCHEDULER_TYPE == "cosine_with_warmup":
                        scheduler.step()
                    elif scheduler and self.args.LR_SCHEDULER_TYPE == "plateau":
                        scheduler.step(avg_val_loss)

                    if early_stopping(avg_val_loss, self):
                        print(f"Early stopping at epoch {epoch+1} for COOP classifier {k}!")
                        break

            if early_stopping.early_stop and early_stopping.best_model_state:
                self.load_state_dict(early_stopping.best_model_state)
                print(f"Loaded best model state for COOP classifier {k} based on early stopping.")
            else:
                 print(f"Training of COOP classifier {k} finished without early stopping.")

            for p in self.coop_classifiers[k].parameters():
                p.requires_grad = False
class DECOOPInferenceEngine:
    def __init__(self, model, eci_thresholds):
        self.model = model
        self.eci_thresholds = eci_thresholds

    def calibrate_q_hat(self, calibration_dataset, alpha=0.1): # Renamed alpha_cp to alpha for consistency with utils_mlp/model.py eci_calibration
        dataloader = DataLoader(calibration_dataset, batch_size=self.model.args.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

        all_non_conformity_scores_val = []

        base_class_set = self.model.base_global_set

        if len(calibration_dataset) == 0:
            print("Validation set empty, fallback q_hat used.")
            q_hat = 0.9
        else:
            with torch.no_grad():
                for batch_data in dataloader:
                    # Note: predict method now requires 'raw_text' in batch_data
                    _, batch_prob_vectors = self.predict(batch_data) # This predict will use the new ZS classifier

                    batch_global_labels = batch_data['global_labels'].cpu().numpy()

                    for b in range(batch_prob_vectors.shape[0]):
                        sample_global_label = batch_global_labels[b].item()
                        sample_probas = batch_prob_vectors[b]

                        if sample_global_label not in base_class_set:
                            continue

                        try:
                            # 确保 sample_global_label 在 sample_probas 的有效索引范围内
                            if sample_global_label >= len(sample_probas): # Changed shape[0] to len(sample_probas) for 1D array
                                print(f"Warning: Global label {sample_global_label} out of bounds for probas shape {sample_probas.shape}. Using score 1.0.")
                                score = 1.0
                            else:
                                prob_true_class = sample_probas[sample_global_label]
                                score = 1.0 - prob_true_class
                        except IndexError:
                            print(f"Warning: IndexError during score calculation for global label {sample_global_label}. Probas shape: {sample_probas.shape}")
                            score = 1.0

                        all_non_conformity_scores_val.append(score)

            all_non_conformity_scores_val = np.array(all_non_conformity_scores_val)
            n_val = len(all_non_conformity_scores_val)
            q_level = np.ceil((n_val + 1) * (1 - alpha)) / n_val
            # Handle empty scores array case
            q_hat = np.quantile(all_non_conformity_scores_val, q_level) if n_val > 0 else 0.9

        self.model.args.CP_OOD_THRESHOLD = q_hat
        print(f"Validation complete. q_hat = {q_hat:.4f}")

    def predict(self, batch):
        self.model.eval()

        # Features now contain the pre-fused (textual + numerical) embeddings
        features = batch["features"].to(self.model.device)
        # raw_texts is NOT needed for MLP-based zs_classifier_
        # raw_texts = batch["raw_text"] # Removed
        raw_texts = batch["raw_text"] 

        B = features.size(0)

        pred_types = [""] * B
        prob_vectors = torch.zeros(B, self.model.num_all_classes, device=self.model.device)

        with torch.no_grad():
            logits_ood_batch = self.model.forward_ood_batch_mlp(features)
            probs_ood_batch = softmax(logits_ood_batch, dim=1)

            max_probs, pred_bases_local_idx = probs_ood_batch.max(dim=1)
            non_conf = 1.0 - max_probs

            best_k_indices = torch.argmin(non_conf, dim=1)

            is_new_mask = torch.zeros(B, dtype=torch.bool, device=self.model.device)

            for b in range(B):
                sample_best_k = best_k_indices[b].item()

                sample_pred_base_local_idx = pred_bases_local_idx[b, sample_best_k].item()
                sample_best_non_conf = float(non_conf[b, sample_best_k])

                sample_ecii_thresh = self.eci_thresholds[sample_best_k].get(sample_pred_base_local_idx, 1.0)
                sample_cp_thresh = getattr(self.model.args, "CP_OOD_THRESHOLD", 1.0)

                is_new_sample = (sample_best_non_conf > sample_ecii_thresh) or (sample_best_non_conf > sample_cp_thresh)

                # Assign directly to the pre-allocated list slot
                pred_types[b] = "NEW" if is_new_sample else "ID"
                is_new_mask[b] = torch.tensor(is_new_sample, dtype=torch.bool, device=self.model.device)
            
            if is_new_mask.any():
                new_idxs = torch.where(is_new_mask)[0].tolist()
                new_texts = [raw_texts[i] for i in new_idxs]
                #   ── Call MLM Zero‑shot classifier on raw texts ──
                zs_probs_np, _ = self.model.zs_classifier_.predict(new_texts)  # returns numpy or torch on CPU
                zs_probs = torch.as_tensor(zs_probs_np, device=self.model.device, dtype=torch.float32)  # (B_new, C_all)
                # Write the probabilities back to the global prob_vectors tensor
                prob_vectors[new_idxs] = zs_probs

            if (~is_new_mask).any():
                id_sample_indices = torch.where(~is_new_mask)[0]

                logits_coop_all_k = self.model.forward_coop_batch_mlp(features[id_sample_indices])

                best_k_for_id_samples = best_k_indices[id_sample_indices]

                best_k_expanded = best_k_for_id_samples.view(-1, 1, 1).expand(-1, logits_coop_all_k.size(1), -1)

                selected_coop_logits = torch.gather(logits_coop_all_k, dim=2, index=best_k_expanded).squeeze(2)

                probs_id_batch = softmax(selected_coop_logits, dim=1)

                global_indices_of_base_classes = torch.tensor(
                    sorted(list(self.model.base_global_set)),
                    dtype=torch.long, device=self.model.device
                )

                prob_vectors[id_sample_indices[:, None], global_indices_of_base_classes] = probs_id_batch.to(self.model.device)

        return pred_types, prob_vectors.cpu().numpy()


    def predict_batch(self, dataset):
        """
        Performs batch inference on the given dataset and collects all prediction results.
        """
        print(f"\nPredicting on Test set with Conformal Prediction...")

        test_batch_size = self.model.args.BATCH_SIZE
        test_dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=16)

        all_point_preds_global = []
        all_prob_matrix_all_classes = []
        all_predictions_conformal_sets = []

        for batch_idx, batch_data in enumerate(test_dataloader):
            # batch_data will now contain 'features' (fused) and 'global_labels'
            # 'raw_texts' is no longer expected as part of batch_data for MLP pipeline
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