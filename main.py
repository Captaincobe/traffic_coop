# main_mlp.py (Modified)

import os
import pandas as pd
import torch
# from datasets.preprocess import Z_Scaler
from datasetsM.preprocess import Z_Scaler
from utils.load_data import _get_features_to_standardize_for_loading, loadData # Import the refactored data loading function
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from args import parameter_parser # Import argument parser
from utils.model import DECOOPInferenceEngine, LLMTrafficDECOOP # Import the refactored model and inference engine

# Import necessary components for feature fusion
from transformers import AutoTokenizer, AutoModel # For LLM embedding
# from datasets.preprocess import Z_Scaler, _get_features_to_standardize_for_loading as get_numerical_features_config
from utils.Dataloader import convert_feature_to_prompt_text


# Suppress tokenizers parallelism warning if it appears (less relevant for MLP but harmless)
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Parse command line arguments and configuration
args = parameter_parser()
DEVICE = args.DEVICE
ALPHA_CP = args.ALPHA_CP
dataset_name = args.dataset_name

# Define all possible traffic labels and their global integer mappings for different datasets
# This is crucial for consistent label handling across the entire pipeline.
if dataset_name == "ISCXVPN2016":
    all_labels_list = ["VPN-MAIL", "VPN-STREAMING", "VPN-VOIP", "VPN-BROWSING","CHAT","STREAMING","MAIL","FT","VPN-FT", "P2P", "BROWSING", "VOIP", "VPN-P2P","VPN-CHAT"]
    ood_labels_for_this_run = ["VOIP"] # Example OOD class for this specific run
elif dataset_name == "ISCXTor2016":
    all_labels_list = ['AUDIO', 'BROWSING', 'CHAT', 'FILE-TRANSFER', 'MAIL', 'P2P', 'VIDEO', 'VOIP']
    ood_labels_for_this_run = ["VOIP"] # Example OOD class for this specific run
# Add other dataset configurations here if needed

# Create a global mapping from string labels to unique integer IDs (0 to C-1)
# This mapping is used universally across data loading, model, and evaluation.
all_class_labels_global_map = {label: i for i, label in enumerate(all_labels_list)}

print(f"Using device: {DEVICE}")

# --- NEW: Pre-fuse textual and numerical features using LLM embeddings ---
# This function is defined inline for demonstration but could be a separate utility.
def fuse_features_and_cache(args, all_labels_list, all_class_labels_global_map, ood_labels_to_exclude, dataset_name):

    # Get paths
    dataset_root = '/home/icdm/code/trafficCOOP/datasetsM'
    csv_path_raw = os.path.join(dataset_root, dataset_name, 'raw',f"{dataset_name}.csv") # Original raw CSV

    # unmapped = [l for l in pd.read_csv(csv_path_raw)['label'].unique()
    #             if l not in all_class_labels_global_map]
    # print(f"Unmapped labels: {unmapped}")
    # Define cache file for fused features
    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"
    fused_cache_file_name = f"fused_features_for_mlp_{ood_suffix}_few_shot_{args.SAMPLES_PER_CLASS}.pt"
    fused_cache_file = os.path.join(dataset_root, dataset_name, "raw", fused_cache_file_name)

    if os.path.exists(fused_cache_file):
        print(f"Loading pre-fused features from cache: {fused_cache_file}")
        return torch.load(fused_cache_file)
    
    # Load tokenizer and LLM for generating textual embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_MODEL_NAME)
    llm_model = AutoModel.from_pretrained(args.LLM_MODEL_NAME).to(args.DEVICE)
    llm_model.eval() # Set LLM to evaluation mode, as we're only extracting features
    print("\n--- Starting Pre-fusion of Textual and Numerical Features ---")

    df_raw = pd.read_csv(csv_path_raw)
    numerical_feature_cols = _get_features_to_standardize_for_loading(dataset_name)

    # Fit numerical scaler on the full raw dataset
    numerical_scaler = Z_Scaler()
    numerical_scaler.fit_transform(df_raw[numerical_feature_cols].copy())

    fused_data_list = []
    
    llm_embedding_dim = llm_model.config.hidden_size 
    numerical_output_dim = len(numerical_feature_cols) 

    print(f"LLM embedding dimension: {llm_embedding_dim}")
    print(f"Numerical features dimension: {numerical_output_dim}")
    print(f"Total fused feature dimension for MLP: {llm_embedding_dim + numerical_output_dim}")

    for idx, row in df_raw.iterrows():
        global_label_str = row['label']
        global_label_numerical = all_class_labels_global_map.get(global_label_str)
        if global_label_numerical is None:
            continue # Skip if label not in map

        # 1. Get Textual Embedding from LLM
        feature_text = convert_feature_to_prompt_text(dataset_name, row)
        full_input_text = f"Traffic classification sample: {feature_text}"
        
        tokenized_input = tokenizer(
            full_input_text,
            truncation=True,
            max_length=args.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(args.DEVICE)

        with torch.no_grad():
            outputs = llm_model(**tokenized_input)
            textual_embedding = outputs.last_hidden_state[:, 0].squeeze(0).cpu() # (H,)

        # 2. Get Scaled Numerical Features
        numerical_features_raw = row[numerical_feature_cols].values
        # numerical_features_scaled = numerical_scaler.fit_transform(numerical_features_raw.reshape(1, -1)).squeeze(0)
        numerical_features_scaled = numerical_scaler.fit_transform(
            row[numerical_feature_cols].values.reshape(1, -1)
        ).squeeze(0)
        numerical_features_tensor = torch.tensor(numerical_features_scaled.astype(float), dtype=torch.float32)
        # 3. Concatenate (Fuse) them
        fused_vector = torch.cat([textual_embedding, numerical_features_tensor], dim=0) # (H + Num_Numerical_Features,)

        fused_data_list.append({
            "fused_features": fused_vector,
            "global_labels": torch.tensor(global_label_numerical, dtype=torch.long)
        })
    
    torch.save(fused_data_list, fused_cache_file)
    print(f"Pre-fused features saved to: {fused_cache_file}")
    print(f"Total fused samples: {len(fused_data_list)}")
    
    return fused_data_list

# Call the feature fusion function
fused_data = fuse_features_and_cache(args, all_labels_list, all_class_labels_global_map, ood_labels_for_this_run, dataset_name)


# Load and preprocess data using the modified loadData that accepts pre-fused data.
# Note: The `args.INPUT_DIM` will be updated inside loadData based on fused_data_list.
train_dataset, val_dataset, test_dataset, args, base_class_indices_num_sorted = loadData(
    args, all_labels_list, all_class_labels_global_map, ood_labels_to_exclude=ood_labels_for_this_run, prefused_data_list=fused_data)

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)} samples")
print(f"Validation dataset size: {len(val_dataset)} samples")
print(f"Test dataset size: {len(test_dataset)} samples")

# Subsetting the training dataset for faster debugging or specific few-shot scenarios (optional)
# The original code includes this, so keeping it for consistency.
indices = torch.randperm(len(train_dataset))[:4096] # Randomly select 4096 samples
train_dataset = torch.utils.data.Subset(train_dataset, indices)
print(f"Training on a subset of {len(train_dataset)} samples.")

# Display key model configuration parameters
NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
print(f"Input features dimension for MLP (after fusion): {args.INPUT_DIM}") # Now reflects fused dim
print(f"Number of base classes: {NUM_BASE_CLASSES}, Total classes: {NUM_ALL_CLASSES}")
print(f"Base class global indices for the model: {base_class_indices_num_sorted}")

print("\nInitializing and Training / Loading MLP-TrafficDECOOP model...")
# Initialize the MLP-based DECOOP model using the updated args.
# LLMTrafficDECOOP in utils_mlp/model.py is essentially an MLPClassifier for each detector.
model_instance = LLMTrafficDECOOP(args) 
# Set the initial global base class indices for the model.
model_instance.set_base_class_global_indices(base_class_indices_num_sorted)

# Define the path where the trained model checkpoint will be saved/loaded
model_save_path = f"./models/{dataset_name}/trained_{args.dataset_name}_{args.SAMPLES_PER_CLASS}_fused_mlp.pth" # Added _fused_mlp to differentiate

if __name__ == "__main__":
    # Check if a trained model exists to load it, otherwise start training
    if os.path.exists(model_save_path):
        print(f"Loading trained model from {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=DEVICE) # Map to current device
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.to(DEVICE)
        model_instance.eval() # Set model to evaluation mode
        # Restore the model's base class global indices to the state it was saved with
        model_instance.set_base_class_global_indices(list(model_instance.base_class_global_indices)) # Ensure it's a list if set uses it that way
        model_instance.eci_thresholds = checkpoint['eci_thresholds'] 
        print("Model and ECI thresholds loaded successfully.")
    else:
        # If no saved model is found, proceed with training
        print("No saved model found. Starting training...")

        # Get a mutable copy of base class indices for shuffling for detector training
        original_base_class_global_indices = list(base_class_indices_num_sorted)
        # Shuffle these to ensure different "pseudo-OOD" classes are chosen for each detector's training
        np.random.shuffle(original_base_class_global_indices)
        
        # Adjust K_DETECTORS if it exceeds the number of available base classes for the leave-one-out setup
        if args.K_DETECTORS > len(original_base_class_global_indices):
            print(f"Warning: K_DETECTORS ({args.K_DETECTORS}) is greater than the number of base classes ({len(original_base_class_global_indices)}). Adjusting K_DETECTORS to {len(original_base_class_global_indices)} for leave-one-class setup.")
            args.K_DETECTORS = len(original_base_class_global_indices)

        # Store the model's overall base class configuration before detector-specific training
        # This will be restored after all detector training phases are complete.
        overall_base_global_indices_state = list(model_instance.base_class_global_indices)

        # --- Phase 1: Train K OOD Detectors ---
        # Each detector `k` is trained with one base class temporarily held out as "pseudo-OOD".
        for k in range(args.K_DETECTORS):
            # The class that will be temporarily considered OOD for detector `k`
            pseudo_ood_class_global_idx = original_base_class_global_indices[k % len(original_base_class_global_indices)]
            
            # Define the ID classes for *this* detector's training (all original base classes except the pseudo-OOD one)
            id_classes_for_detector_k_global_indices = [
                idx for idx in original_base_class_global_indices if idx != pseudo_ood_class_global_idx
            ]

            # Temporarily set the model's base class configuration for the current detector's training and calibration.
            # This makes `model.fit` and `model.eci_calibration` identify ID/OOD samples correctly for this specific detector.
            model_instance.set_base_class_global_indices(id_classes_for_detector_k_global_indices)
            
            print(f"\n[Training OOD Detector #{k}] Pseudo-OOD class for this detector: {pseudo_ood_class_global_idx}")
            print(f"  ID classes for this detector: {id_classes_for_detector_k_global_indices}")
            
            # Train the k-th OOD detector using the specified training dataset.
            # The `fit` method uses the `model_instance.base_global_set` to differentiate ID from pseudo-OOD.
            print(f"\n[Training OOD Detector #{k}] with {len(train_dataset)} samples (ID and pseudo-OOD determined internally).")
            model_instance.fit(train_dataset, val_dataset, k=k)
            
            # Calibrate the ECI non-conformity thresholds for the k-th OOD detector.
            model_instance.eci_calibration(val_dataset, k=k, alpha=ALPHA_CP)

        # Restore the model's base class configuration to the overall set of base classes after detector training
        # This is important for the COOP classifier training and final inference.
        model_instance.set_base_class_global_indices(overall_base_global_indices_state)
        print(f"\nRestored model's base class configuration to: {model_instance.base_class_global_indices}")

        # --- Phase 2: Train K COOP Classifiers ---
        # These classifiers are trained using a mix of ID samples (classified by the OOD detectors)
        # and OOD samples (whose knowledge is distilled from the ZS classifier).
        model_instance.fit_sub_classifiers(train_dataset, val_dataset)
        print("MLP-TrafficDECOOP Model Training Finished.")

        # try:
        #     # Save the trained model's state dictionary and the calibrated ECI thresholds
        #     torch.save({
        #         'model_state_dict': model_instance.state_dict(),
        #         'eci_thresholds': model_instance.eci_thresholds
        #     }, model_save_path)
        #     print(f"Model and ECI thresholds saved to: {model_save_path}")
        # except Exception as e:
        #     print(f"Error saving model: {e}")

    # ----------------- Conformal Prediction Calibration and Inference -----------------
    # Initialize the DECOOP inference engine with the trained model and its ECI thresholds.
    engine = DECOOPInferenceEngine(model_instance, eci_thresholds=model_instance.eci_thresholds)

    print("\nStarting Global Conformal Prediction (CP) Calibration...")
    # Calibrate the global CP OOD threshold (q_hat) using the validation set.
    # This threshold is used in the final decision for "NEW" vs. "ID" in the `predict` method.
    engine.calibrate_q_hat(val_dataset, args.ALPHA_CP)

    print("\nPerforming Inference on Test Set with Conformal Prediction...")
    # Perform batch predictions on the test dataset to get point predictions, probability matrix,
    # and conformal sets.
    point_preds_global, prob_matrix_all_classes, predictions_conformal_sets = \
        engine.predict_batch(test_dataset)

    # Extract true global labels from the test dataset for evaluation purposes.
    y_test_true_global = np.array([sample['global_labels'].item() for sample in test_dataset]) 


    # ----------------- Evaluation Metrics -----------------
    print("\n--- Evaluating Conformal Prediction Performance ---")
    coverage = 0
    avg_set_size = 0
    # Calculate coverage and average set size for conformal prediction
    for i in range(len(test_dataset)):
        true_label = y_test_true_global[i]
        # Check if the true label is contained within the predicted conformal set
        if true_label in predictions_conformal_sets[i]:
            coverage += 1
        avg_set_size += len(predictions_conformal_sets[i])
    actual_coverage = coverage / len(test_dataset)
    avg_set_size = avg_set_size / len(test_dataset)

    print(f"Conformal Prediction Target Coverage (1-alpha): {1 - ALPHA_CP:.4f}")
    print(f"Actual Conformal Prediction Coverage: {actual_coverage:.4f}")
    print(f"Average Conformal Set Size: {avg_set_size:.2f}")

    print("\n--- Evaluating Point Prediction Performance ---")
    # Calculate overall accuracy of the point predictions
    acc = np.mean(np.array(point_preds_global) == y_test_true_global)
    print(f"Overall Accuracy: {acc:.4f}")

    # ----------- Base / New Accuracy -----------
    # Build string lists for base (seen) and new (unseen) classes
    base_traffic_labels_str = [
        lbl for lbl, idx in all_class_labels_global_map.items()
        if idx in base_class_indices_num_sorted
    ]
    new_traffic_labels_str = [
        lbl for lbl in all_labels_list
        if lbl not in base_traffic_labels_str
    ]
    Y_b_indices = [all_class_labels_global_map[l] for l in base_traffic_labels_str]
    Y_n_indices = [all_class_labels_global_map[l] for l in new_traffic_labels_str]

    mask_base = np.isin(y_test_true_global, Y_b_indices)
    mask_new  = np.isin(y_test_true_global, Y_n_indices)

    acc_base = (point_preds_global[mask_base] == y_test_true_global[mask_base]).mean() if mask_base.any() else np.nan
    acc_new  = (point_preds_global[mask_new]  == y_test_true_global[mask_new]).mean() if mask_new.any() else np.nan
    harmonic = 2*acc_base*acc_new / (acc_base+acc_new+1e-8)

    # Generate a detailed classification report
    label_map = {v: k for k, v in all_class_labels_global_map.items()} # Reverse map for report names
    # Identify all labels present in the true and predicted sets for the report
    report_labels = sorted(list(set(y_test_true_global) | set(point_preds_global)))
    target_names = [label_map.get(l, f"Class_{l}") for l in report_labels]

    try:
        print(classification_report(
            y_test_true_global, point_preds_global, labels=report_labels,
            target_names=target_names, zero_division=0, digits=4
        ))
    except Exception as e:
        print(f"Failed to generate classification report: {e}. Error: {e}")

    # Calculate AUROC (Area Under the Receiver Operating Characteristic Curve)
    try:
        # Binarize true labels for multi-class AUROC calculation
        y_bin = label_binarize(y_test_true_global, classes=list(range(NUM_ALL_CLASSES)))
        if NUM_ALL_CLASSES == 2:
            # For binary classification, AUROC is calculated directly from probabilities of the positive class
            auc = roc_auc_score(y_test_true_global, prob_matrix_all_classes[:, 1])
        else:
            # For multi-class, use 'ovr' (one-vs-rest) strategy and weighted averaging
            auc = roc_auc_score(y_bin, prob_matrix_all_classes, multi_class='ovr', average='weighted')
        print(f"Weighted AUROC: {auc:.4f}")
        print(f"\nAcc_base={acc_base:.4f} | Acc_new={acc_new:.4f} | H-mean={harmonic:.4f}")
    except Exception as e:
        print(f"AUROC calculation error: {e}. Ensure enough classes are present for multi-class AUROC.")