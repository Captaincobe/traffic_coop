import os
import torch
from utils.load_data import loadData
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import torchvision
from args import parameter_parser
from utils.model import DECOOPInferenceEngine, LLMTrafficDECOOP
torchvision.disable_beta_transforms_warning()

args = parameter_parser()
DEVICE = args.DEVICE
ALPHA_CP = args.ALPHA_CP
dataset_name = args.dataset_name


if dataset_name == "ISCXVPN2016":
    all_labels_list = ["VPN-MAIL", "VPN-STREAMING", "VPN-VOIP", "VPN-BROWSING","CHAT","STREAMING","MAIL","FT","VPN-FT", "P2P", "BROWSING", "VOIP", "VPN-P2P","VPN-CHAT"]
    ood_labels_for_this_run = ["VOIP"] 
    base_traffic_labels_str = all_labels_list
    args.LABEL_TOKEN_MAP={"BROWSING": "web",         
        "CHAT": "chat",           
        "VOIP": "call",            
        "P2P": "sharing",          
        "MAIL": "mail",          
        "STREAMING": "stream",      
        "VPN-BROWSING": "vpnweb",  
        "VPN-CHAT": "vpnchat",     
        "VPN-FT": "vpnftp",        
        "VPN-MAIL": "vpnmail",    
        "VPN-P2P": "vpnshare",     
        "VPN-STREAMING": "vpnvideo", 
        "VPN-VOIP": "vpncall",     
        "FT": "ftp"}
elif dataset_name == "ISCXTor2016":
    all_labels_list = ['AUDIO', 'BROWSING', 'CHAT', 'FILE-TRANSFER', 'MAIL', 'P2P', 'VIDEO', 'VOIP']
    ood_labels_for_this_run = ["VOIP"] 
    base_traffic_labels_str = all_labels_list
    args.LABEL_TOKEN_MAP={"AUDIO": "audio",         
        "VIDEO": "video",         
        "FILE-TRANSFER": "file",         
        "BROWSING": "web",         
        "CHAT": "chat",           
        "VOIP": "call",            
        "P2P": "sharing",          
        "MAIL": "mail"}

# 给 每个 标签一个唯一数字 0‒(C-1)，无论它是基类还新类 整个项目只此一份；一旦确定就不会再变
all_class_labels_global_map = {label: i for i, label in enumerate(all_labels_list)}

print(f"Device: {DEVICE}")

train_dataset, val_dataset, test_dataset, args, base_class_indices_num_sorted = loadData(
    args, base_traffic_labels_str, all_class_labels_global_map, ood_labels_to_exclude=ood_labels_for_this_run)
print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
# a part of sample
indices = torch.randperm(len(train_dataset))[:4096]
train_dataset_subset = torch.utils.data.Subset(train_dataset, indices)
train_dataset = train_dataset_subset

NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
print(f"Input textualized, max_seq_len: {args.MAX_SEQ_LENGTH}")
print(f"Number of base classes: {NUM_BASE_CLASSES}, Total classes: {NUM_ALL_CLASSES}")
print(f"Base class global indices: {base_class_indices_num_sorted}")

print("\nInitializing and Training model...")
model_instance = LLMTrafficDECOOP(args)
model_instance.set_base_class_global_indices(base_class_indices_num_sorted)
model_save_path = f"./models/{dataset_name}/trained_{args.dataset_name}_{args.SAMPLES_PER_CLASS}.pth"

# debug_model_training(train_dataset, model_instance, base_class_indices_num_sorted)

if __name__ == "__main__":
    if os.path.exists(model_save_path):
        model_instance.load_state_dict(torch.load(model_save_path))
        model_instance.to(DEVICE)
        model_instance.eval()
        original_model_base_global_indices_state = list(model_instance.base_class_global_indices)
        model_instance.set_base_class_global_indices(original_model_base_global_indices_state)
    else:
        non_conformity_scores_val = []

        original_base_class_global_indices = list(base_class_indices_num_sorted) #
        np.random.shuffle(original_base_class_global_indices)
        
        if args.K_DETECTORS > len(original_base_class_global_indices):
            print(f"Warning: K_DETECTORS ({args.K_DETECTORS}) is greater than the number of base classes ({len(original_base_class_global_indices)}). Adjusting K_DETECTORS to {len(original_base_class_global_indices)} for leave-one-class setup.")
            args.K_DETECTORS = len(original_base_class_global_indices)

        original_model_base_global_indices_state = list(model_instance.base_class_global_indices)

        for k in range(args.K_DETECTORS):
            # The class that will be temporarily considered OOD for detector 'k'
            # We cycle through the original base classes to assign a unique pseudo-OOD for each detector
            pseudo_ood_class_global_idx = original_base_class_global_indices[k % len(original_base_class_global_indices)]
            
            # Define the ID classes for *this* detector's training (all original base classes except the pseudo-OOD one)
            id_classes_for_detector_k_global_indices = [
                idx for idx in original_base_class_global_indices if idx != pseudo_ood_class_global_idx
            ]

            # Temporarily set the model's base class configuration for the current detector's training/calibration phase
            # This is crucial for `model.fit` and `model.eci_calibration` to correctly identify ID/OOD.
            model_instance.set_base_class_global_indices(id_classes_for_detector_k_global_indices)
            
            print(f"\n[Training PromptLearner OOD #{k}] Pseudo-OOD class for this detector: {pseudo_ood_class_global_idx}")
            print(f"  ID classes for this detector: {id_classes_for_detector_k_global_indices}")
            # Pass the full train_dataset (which contains all original base classes) to the training function.
            # The `fit` method will use the temporarily set `model_instance.base_global_set`
            # to distinguish ID samples from the pseudo-OOD samples (the left-out class).
            print(f"\n[Training PromptLearner OOD #{k}] with {len(train_dataset)} samples (ID and pseudo-OOD determined internally).")
            model_instance.fit(train_dataset, k=k)
            
            # Similarly, pass the full val_dataset for calibration.
            # `eci_calibration` will also use the temporarily set `model_instance.base_global_set`
            # to collect non-conformity scores only for the ID samples.
            model_instance.eci_calibration(val_dataset, k=k, alpha=ALPHA_CP)

        model_instance.set_base_class_global_indices(original_model_base_global_indices_state)
        print(f"\nRestored model's base class configuration to: {model_instance.base_class_global_indices}")

        model_instance.fit_sub_classifiers(train_dataset)
        print("LLMTrafficDECOOP Model Training Finished.")

        try:
            torch.save(model_instance.state_dict(), model_save_path)
        except Exception as e:
            print(f"Error saving model: {e}")


    # ----------------- Conformal Calibration -----------------
    engine = DECOOPInferenceEngine(model_instance, eci_thresholds=model_instance.eci_thresholds)

    print("\nStarting Conformal Prediction Validation...")

    # 调用 engine 的 calibrate_q_hat 方法进行校准
    engine.calibrate_q_hat(val_dataset, args.ALPHA_CP)

    point_preds_global, prob_matrix_all_classes, predictions_conformal_sets = \
        engine.predict_batch(test_dataset)

    # 从 test_dataset 中预先提取所有真实标签，用于最终评估
    y_test_true_global = np.array([s['global_labels'].item() for s in test_dataset]) 


    # ----------------- Evaluatfion -----------------
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



