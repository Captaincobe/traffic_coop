import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import torchvision

from args import parameter_parser
from models.model import LLMTrafficDECOOP
from utils.load_data import load_and_prepare_data_for_llm
torchvision.disable_beta_transforms_warning()

# --- 0. Configuration & Hyperparameters ---
args = parameter_parser()
LLM_MODEL_NAME = args.LLM_MODEL_NAME
DEVICE = args.DEVICE
MAX_SEQ_LENGTH = args.MAX_SEQ_LENGTH
NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
ALPHA_CP = args.ALPHA_CP
dataset_name = args.dataset_name
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
except Exception as e:
    print(f"Error loading tokenizer for {LLM_MODEL_NAME}: {e}")
    exit()


# --- Main Execution ---
if __name__ == "__main__":
    if dataset_name == "ISCXVPN2016":
        base_traffic_labels_str = ["VPN-MAIL", "VPN-STREAMING", "VPN-VOIP", "BROWSING","CHAT","STREAMING","MAIL","FT","VPN-FT", "VPN-P2P"]
        new_traffic_labels_str = ["VPN-BROWSING", "VOIP", "P2P","VPN-CHAT"] 

    all_labels_list = sorted(list(set(base_traffic_labels_str + new_traffic_labels_str)))
    all_class_labels_global_map = {label: i for i, label in enumerate(all_labels_list)}
    
    print(f"Device: {DEVICE}")
    print("Loading and preparing data for LLM...")
    train_dataset, val_dataset, test_dataset, args, _, base_class_indices_num_sorted = load_and_prepare_data_for_llm(
        args,
        base_traffic_labels_str,
        all_class_labels_global_map,
        tokenizer, # Globally defined tokenizer
    )

    NUM_BASE_CLASSES = args.NUM_BASE_CLASSES
    NUM_ALL_CLASSES = args.NUM_ALL_CLASSES
    print(f"Input textualized, max_seq_len: {args.MAX_SEQ_LENGTH}")
    print(f"Number of base classes (for model training): {NUM_BASE_CLASSES}")
    print(f"Total number of unique classes (for output vector): {NUM_ALL_CLASSES}")
    print(f"Base class global indices used for mapping: {base_class_indices_num_sorted}")

    print(f"MODEL training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    if NUM_BASE_CLASSES <=0 or NUM_ALL_CLASSES <=0:
        raise ValueError("Data loading failed to set global class dimensions properly.")
    if len(train_dataset) == 0:
        raise ValueError("MODEL training dataset is empty. Check base class definitions and data.")

    print("\nInitializing and Training model...")
    model_instance = LLMTrafficDECOOP(args)
    model_instance.set_base_class_global_indices(base_class_indices_num_sorted) # check baseclass
    model_instance.fit(train_dataset) # train_dataset uses local labels
    print("LLMTrafficDECOOP Model Training Finished.")
    # 使用验证集校准 ECI 阈值
    base_class_set_for_calibration = set(base_class_indices_num_sorted)
    model_instance.calibrate_eci_threshold(val_dataset, base_class_set_for_calibration)
    
    # --- Conformal Prediction Validation (uses global labels) ---
    print("\nStarting Conformal Prediction Validation...")
    non_conformity_scores_val = []
    if len(val_dataset) == 0:
        print("Warning: validation dataset is empty. Using a default q_hat. CP results will not be meaningful.")
        q_hat = 0.9 
    else:
        for i in range(len(val_dataset)):
            tokenized_input_dict = val_dataset[i] # Dataset __getitem__ returns dict
            y_c_true_global_numerical = tokenized_input_dict['global_labels'].item() # Get global label
            
            probas_x_c_all_classes = model_instance.predict_proba(tokenized_input_dict) 
            
            if 0 <= y_c_true_global_numerical < len(probas_x_c_all_classes):
                 prob_true_label = probas_x_c_all_classes[y_c_true_global_numerical]
                 score = 1.0 - prob_true_label
            else:
                 print(f"Warning: True label {y_c_true_global_numerical} out of bounds for probas vector of length {len(probas_x_c_all_classes)}")
                 score = 1.0 
            non_conformity_scores_val.append(score)
        non_conformity_scores_val = np.array(non_conformity_scores_val)
        n_val = len(val_dataset)
        q_level = np.ceil((n_val + 1) * (1 - ALPHA_CP)) / n_val
        q_level = min(max(q_level, 0.0), 1.0) 
        q_hat = np.quantile(non_conformity_scores_val, q_level) if n_val > 0 else 0.9 # fallback q_hat if n_val is 0
    
    print(f"Validation complete. q_hat = {q_hat:.4f}")
    
    # --- Evaluation (uses global labels) ---
    # (Using the evaluation logic from your previous `cpcoop.py` which was already quite detailed)
    # Just ensure that the inputs to predict_proba are the tokenized dicts
    print("\nPredicting on test set with Conformal Prediction...")
    predictions_conformal_sets = [] 
    if len(test_dataset) > 0:
        y_test_true_all_global_numerical = np.array([data['global_labels'].item() for data in test_dataset])
        decoop_point_predictions_global_numerical = []
        decoop_probability_scores_for_all_classes = np.zeros((len(test_dataset), NUM_ALL_CLASSES))

        for i in range(len(test_dataset)):
            tokenized_input_dict = test_dataset[i]
            # y_t_true_global_numerical = tokenized_input_dict['global_labels'].item() # Already got all true labels

            probas_all_classes = model_instance.predict_proba(tokenized_input_dict)
            decoop_probability_scores_for_all_classes[i, :] = probas_all_classes
            predicted_global_class_idx = np.argmax(probas_all_classes)
            decoop_point_predictions_global_numerical.append(predicted_global_class_idx)

            current_prediction_set_indices = []
            for c_idx in range(NUM_ALL_CLASSES):
                prob_candidate_class = probas_all_classes[c_idx]
                non_conformity_candidate = 1.0 - prob_candidate_class
                if non_conformity_candidate <= q_hat:
                    current_prediction_set_indices.append(c_idx)
            predictions_conformal_sets.append(current_prediction_set_indices)
        
        # CP Evaluation
        coverage_count = 0; avg_set_size_sum = 0
        for i in range(len(test_dataset)):
            true_label = y_test_true_all_global_numerical[i]
            if true_label in predictions_conformal_sets[i]: coverage_count +=1
            avg_set_size_sum += len(predictions_conformal_sets[i])
        actual_coverage = coverage_count/len(test_dataset) if len(test_dataset) > 0 else 0
        avg_set_size = avg_set_size_sum/len(test_dataset) if len(test_dataset) > 0 else 0
        print(f"Conformal Prediction: Target Coverage: {1-ALPHA_CP:.4f}, Actual: {actual_coverage:.4f}, Avg Set Size: {avg_set_size:.2f}")

        # DECOOP Point Prediction Metrics
        print("\nDECOOP Point Prediction Performance (Global Labels):")
        # Accuracy already calculated if decoop_point_predictions_global_numerical is filled
        accuracy = np.mean(np.array(decoop_point_predictions_global_numerical) == y_test_true_all_global_numerical)
        print(f"  Accuracy: {accuracy:.4f}")

        report_labels_global = sorted(list(set(y_test_true_all_global_numerical) | set(decoop_point_predictions_global_numerical)))
        idx_to_label_str_map = {v: k for k, v in all_class_labels_global_map.items()} # For target names
        target_names_report_global = [idx_to_label_str_map.get(l, f"Class_{l}") for l in report_labels_global]
        
        if not target_names_report_global and report_labels_global : target_names_report_global = [f"Class_{idx}" for idx in report_labels_global]

        try:
            class_report = classification_report(
                y_test_true_all_global_numerical, decoop_point_predictions_global_numerical,
                labels=report_labels_global, target_names=target_names_report_global,
                zero_division=0, digits=4 )
            print("  Classification Report:")
            print(class_report)
        except Exception as e: print(f"Could not generate classification report: {e}")

        if len(np.unique(y_test_true_all_global_numerical)) > 1 and \
           y_test_true_all_global_numerical.shape[0] == decoop_probability_scores_for_all_classes.shape[0] :
            try:
                y_test_binarized_global = label_binarize(y_test_true_all_global_numerical, classes=list(range(NUM_ALL_CLASSES)))
                if NUM_ALL_CLASSES == 2: # Binary
                    auroc_score = roc_auc_score(y_test_true_all_global_numerical, decoop_probability_scores_for_all_classes[:, 1])
                elif y_test_binarized_global.shape[1] > 1: # Multi-class with >1 effective binarized classes
                     auroc_score = roc_auc_score(y_test_binarized_global, decoop_probability_scores_for_all_classes, multi_class='ovr', average='weighted')
                else:
                    auroc_score = np.nan 
                    print("  AUROC not computed: Binarized y_true has only one effective class or NUM_ALL_CLASSES is not suitable for this binarization.")
                if not np.isnan(auroc_score): print(f"  AUROC (Weighted OVR): {auroc_score:.4f}")
            except ValueError as e: print(f"Could not compute AUROC: {e}")
        else: print("  AUROC not computed (single class in y_true or data shape mismatch).")
    else: print("Test dataset is empty. No evaluation performed.")
    