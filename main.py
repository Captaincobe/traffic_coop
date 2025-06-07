import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import torchvision
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F

from args import parameter_parser
from models.model import DECOOPInferenceEngine, LLMTrafficDECOOP
from utils.load_data import load_and_prepare_data_for_llm
torchvision.disable_beta_transforms_warning()
from collections import defaultdict
from sklearn.model_selection import KFold

args = parameter_parser()
LLM_MODEL_NAME = args.LLM_MODEL_NAME
DEVICE = args.DEVICE
NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
ALPHA_CP = args.ALPHA_CP
dataset_name = args.dataset_name

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

if dataset_name == "ISCXVPN2016":
    base_traffic_labels_str = ["VPN-MAIL", "VPN-STREAMING", "VPN-VOIP", "BROWSING","CHAT","STREAMING","MAIL","FT","VPN-FT", "VPN-P2P"]
    new_traffic_labels_str = ["VPN-BROWSING", "VOIP", "P2P","VPN-CHAT"] 

all_labels_list = sorted(list(set(base_traffic_labels_str + new_traffic_labels_str)))
all_class_labels_global_map = {label: i for i, label in enumerate(all_labels_list)}

print(f"Device: {DEVICE}")
train_dataset, val_dataset, test_dataset, args, _, base_class_indices_num_sorted = load_and_prepare_data_for_llm(
    args, base_traffic_labels_str, all_class_labels_global_map, tokenizer
)

NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
print(f"Input textualized, max_seq_len: {args.MAX_SEQ_LENGTH}")
print(f"Number of base classes: {NUM_BASE_CLASSES}, Total classes: {NUM_ALL_CLASSES}")
print(f"Base class global indices: {base_class_indices_num_sorted}")

print("\nInitializing and Training model...")
model_instance = LLMTrafficDECOOP(args)
model_instance.set_base_class_global_indices(base_class_indices_num_sorted)


# K-fold leave-one-class-out split for pseudo-OOD estimation
kf = KFold(n_splits=args.K_DETECTORS, shuffle=True, random_state=42)
base_class_to_indices = defaultdict(list)
for idx, sample in enumerate(train_dataset):
    y = sample['global_labels'].item()
    if y in base_class_indices_num_sorted:
        base_class_to_indices[y].append(idx)

base_class_indices = sorted(base_class_to_indices.keys())
splits = list(kf.split(base_class_indices))

# For collecting entropy scores for CP
non_conformity_scores_val = []


for k, (train_class_idx, val_class_idx) in enumerate(splits):
    train_ids, val_ids = [], []
    for i in train_class_idx:
        train_ids.extend(base_class_to_indices[base_class_indices[i]])
    for j in val_class_idx:
        val_ids.extend(base_class_to_indices[base_class_indices[j]])

    train_subset = torch.utils.data.Subset(train_dataset, train_ids)
    val_subset = torch.utils.data.Subset(train_dataset, val_ids)

    print(f"\n[Training PromptLearner OOD #{k}] with {len(train_ids)} ID and {len(val_ids)} pseudo-OOD samples")
    # Prompt tuning training procedure
    train_loader = DataLoader(train_subset, batch_size=args.BATCH_SIZE, shuffle=True)
    prompt_params = list(model_instance.prompt_manager.ood_prompts[k].parameters())
    optimizer = torch.optim.Adam(prompt_params, lr=args.LEARNING_RATE_PROMPT)

    model_instance.prompt_manager.encoder.eval()  # Freeze encoder
    model_instance.prompt_manager.ood_prompts[k].train()

    for epoch in range(args.NUM_EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            # 将全局 label 映射为当前 fold 的局部 label
            train_class_global_labels = [base_class_indices_num_sorted[i] for i in train_class_idx]
            global_to_local = {g: i for i, g in enumerate(train_class_global_labels)}
            labels = batch['global_labels'].tolist()
            labels = [global_to_local[l] for l in labels]
            labels = torch.tensor(labels).to(DEVICE)

            logits = model_instance.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[PromptTuning][Epoch {epoch+1}] Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[PromptTuning][Epoch {epoch+1}/{args.NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")

    # Evaluate entropy on val_subset to collect for CP
    for sample in val_subset:
        input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model_instance.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
            non_conformity_scores_val.append(entropy)

# 估计 entropy 阈值 τ，可以使用 validation set 平均 entropy
# 示例：固定阈值
tau_entropy = 1.2  # 可调参数

model_instance.fit_sub_classifiers(train_dataset, threshold_tau=tau_entropy)
print("LLMTrafficDECOOP Model Training Finished.")

# ----------------- Conformal Calibration -----------------
print("\nStarting Conformal Prediction Validation...")
base_class_set = set(base_class_indices_num_sorted)
base_class_list = sorted(list(base_class_set))
non_conformity_scores_val = []

if len(val_dataset) == 0:
    print("Validation set empty, fallback q_hat used.")
    q_hat = 0.9
else:
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        y_global = sample['global_labels'].item()
        if y_global not in base_class_set:
            continue
        probas = model_instance.predict_proba(sample)
        try:
            idx = base_class_list.index(y_global)
            score = 1.0 - probas[idx]
        except Exception:
            score = 1.0
        non_conformity_scores_val.append(score)

    non_conformity_scores_val = np.array(non_conformity_scores_val)
    n_val = len(non_conformity_scores_val)
    q_level = np.ceil((n_val + 1) * (1 - ALPHA_CP)) / n_val
    q_hat = np.quantile(non_conformity_scores_val, q_level) if n_val > 0 else 0.9

args.CP_OOD_THRESHOLD = q_hat

print(f"Validation complete. q_hat = {q_hat:.4f}")

# ----------------- Prediction on Test Set -----------------
print("\nPredicting on test set with Conformal Prediction...")
y_test_true_global = np.array([s['global_labels'].item() for s in test_dataset])
point_preds_global = []
prob_matrix_all_classes = np.zeros((len(test_dataset), NUM_ALL_CLASSES))
predictions_conformal_sets = []

for i in range(len(test_dataset)):
    sample = test_dataset[i]
    probas = model_instance.predict_proba(sample)
    base_class_list_sorted = base_class_list

    # argmax over base classes → map to global index
    pred_base_idx = np.argmax(probas)
    pred_global_idx = base_class_list_sorted[pred_base_idx] if pred_base_idx < len(base_class_list_sorted) else -1
    point_preds_global.append(pred_global_idx)

    # fill prob vector
    for j, global_idx in enumerate(base_class_list_sorted):
        prob_matrix_all_classes[i, global_idx] = probas[j]

    # conformal prediction set
    cp_set = []
    for j, global_idx in enumerate(base_class_list_sorted):
        if 1.0 - probas[j] <= q_hat:
            cp_set.append(global_idx)
    predictions_conformal_sets.append(cp_set)

# ----------------- Evaluation -----------------
print("\nEvaluating Conformal Prediction...")
coverage = 0
avg_size = 0
for i in range(len(test_dataset)):
    true_label = y_test_true_global[i]
    if i >= len(predictions_conformal_sets):
        continue
    if true_label in predictions_conformal_sets[i]:
        coverage += 1
    avg_size += len(predictions_conformal_sets[i])
actual_coverage = coverage / len(test_dataset)
avg_set_size = avg_size / len(test_dataset)

print(f"CP Target Coverage: {1 - ALPHA_CP:.4f}, Actual: {actual_coverage:.4f}, Avg Set Size: {avg_set_size:.2f}")

# Classification report
print("\nPoint Prediction Evaluation:")
acc = np.mean(np.array(point_preds_global) == y_test_true_global)
print(f"Accuracy: {acc:.4f}")

label_map = {v: k for k, v in all_class_labels_global_map.items()}
report_labels = sorted(list(set(y_test_true_global) | set(point_preds_global)))
target_names = [label_map.get(l, f"Class_{l}") for l in report_labels]

try:
    print(classification_report(
        y_test_true_global, point_preds_global, labels=report_labels,
        target_names=target_names, zero_division=0, digits=4
    ))
except Exception as e:
    print(f"Failed to generate classification report: {e}")

# AUROC
try:
    y_bin = label_binarize(y_test_true_global, classes=list(range(NUM_ALL_CLASSES)))
    if NUM_ALL_CLASSES == 2:
        auc = roc_auc_score(y_test_true_global, prob_matrix_all_classes[:, 1])
    else:
        auc = roc_auc_score(y_bin, prob_matrix_all_classes, multi_class='ovr', average='weighted')
    print(f"AUROC: {auc:.4f}")
except Exception as e:
    print(f"AUROC error: {e}")

engine = DECOOPInferenceEngine(model_instance, eci_thresholds=None)
