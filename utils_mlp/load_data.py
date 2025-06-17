import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Removed: from transformers import AutoTokenizer

# Import the refactored Dataset class
from utils_mlp.Dataloader import MLPCSVTrafficDataset

# Import convert_feature_to_prompt_text from the LLM utils Dataloader
from utils.Dataloader import convert_feature_to_prompt_text # 新增导入

# Helper function to get the list of features that are expected to be standardized
# and used as input to the MLP. This function replicates the relevant part
# from `datasets/preprocess.py` to ensure consistency in feature selection.
# In a larger project, this would ideally be imported from a shared config module.
def _get_features_to_standardize_for_loading(dataset_name):
    """
    Returns the list of numerical features expected for the given dataset
    after preprocessing and standardization, to be used as MLP input.
    """
    if dataset_name == 'ISCXVPN2016':
        return [
            "duration", "total_fiat", "total_biat",
            "min_fiat", "min_biat", "max_fiat", "max_biat",
            "mean_fiat", "mean_biat", "std_fiat", "std_biat",
            "flowPktsPerSecond", "flowBytesPerSecond",
            "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
            "min_active", "mean_active", "max_active", "std_active",
            "min_idle", "mean_idle", "max_idle", "std_idle"
        ]
    elif dataset_name == 'ISCXTor2016':
        # The original `preprocess.py` does not explicitly define FEATURES_TO_STANDARDIZE
        # for 'ISCXTor2016'. Based on the structure and common traffic features,
        # these are assumed to be the numerical features after preprocessing.
        return [
            "Flow Duration", "Flow Packets/s", "Flow Bytes/s", "Fwd IAT Mean",
            "Fwd IAT Min", "Bwd IAT Mean", "Bwd IAT Min", "Active Mean", "Idle Mean"
        ]
    elif dataset_name == 'TONIoT':
        return [
            'proto', 'service', 'duration', 'src_bytes', 'dst_bytes', 'conn_state',
            'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'dns_query', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected',
            'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established', 'ssl_subject',
            'ssl_issuer', 'http_trans_depth', 'http_method', 'http_uri', 'http_referrer',
            'http_version', 'http_request_body_len', 'http_response_body_len',
            'http_status_code', 'http_user_agent', 'http_orig_mime_types',
            'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice',
        ]
    elif dataset_name == 'CICIDS':
        return [
            'Protocol','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets',
            'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std',
            'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
            'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
            'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total',
            'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Fwd URG Flags',
            'Fwd Header Length','Bwd Header Length', 'Fwd Packets/s','Bwd Packets/s','Min Packet Length',
            'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance',
            'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count',
            'Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size',
            'Fwd Header Length2',
            'Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward',
            'act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
        ]
    else:
        raise ValueError(f"Dataset '{dataset_name}' not configured for feature extraction.")

def loadData(args, full_traffic_labels_list, all_class_labels_global_map, ood_labels_to_exclude=None):
    """
    Loads, splits, and processes traffic data for MLP training and evaluation.
    Handles caching of processed numerical data.

    Args:
        args: An argparse.Namespace object containing configuration parameters
              like `dataset_name`, `SAMPLES_PER_CLASS`, `DEVICE`, etc.
        full_traffic_labels_list: A list of all possible traffic class labels (strings).
        all_class_labels_global_map: A dictionary mapping string labels to their global integer IDs.
        ood_labels_to_exclude: A list of string labels to be considered OOD for the current run.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, updated_args, base_class_global_indices_sorted)
    """
    dataset_name = args.dataset_name
    num_samples_per_class = args.SAMPLES_PER_CLASS

    if ood_labels_to_exclude is None:
        ood_labels_to_exclude = []

    base_traffic_labels_str = [label for label in full_traffic_labels_list if label not in ood_labels_to_exclude]
    new_traffic_labels_str = [label for label in full_traffic_labels_list if label in ood_labels_to_exclude]
    dataset_root='/home/icdm/code/trafficCOOP/datasets'

    # The CSV path points to the file generated by `datasets/preprocess.py`
    csv_path = os.path.join(dataset_root, dataset_name, "raw", f"{dataset_name}--1.csv")

    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"

    # Changed cache file name to reflect storage of numerical data and LLM ZS usage
    cache_file_name = f"numerical_llm_zs_{ood_suffix}_{num_samples_per_class}.pt" # 更新缓存文件名以反映 LLM ZS 分类器
    cache_file = os.path.join(dataset_root, dataset_name, "raw", cache_file_name)

    # Get the list of feature columns that will be extracted for MLP input
    feature_cols = _get_features_to_standardize_for_loading(dataset_name)

    if os.path.exists(cache_file):
        print(f"Loading numerical datasets from cache: {cache_file}")
        cached = torch.load(cache_file)

        train_dataset = MLPCSVTrafficDataset(cached["train_numerical_list"])
        val_dataset = MLPCSVTrafficDataset(cached["val_numerical_list"])
        test_dataset = MLPCSVTrafficDataset(cached["test_numerical_list"])
        args.NUM_BASE_CLASSES = cached["args"].NUM_BASE_CLASSES
        args.NUM_ALL_CLASSES = cached["args"].NUM_ALL_CLASSES
        args.INPUT_DIM = cached["args"].INPUT_DIM # Load the input dimension
        # Ensure LLM-related args are loaded from cache if needed
        args.LLM_MODEL_NAME = cached["args"].LLM_MODEL_NAME
        args.LABEL_TOKEN_MAP = cached["args"].LABEL_TOKEN_MAP
        base_class_global_indices_sorted = cached["base_class_indices"]
    else:
        df = pd.read_csv(csv_path)

        # Validate that all expected feature columns exist in the loaded DataFrame
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in CSV for {dataset_name}: {missing_cols}. Please ensure `datasets/preprocess.py` has been run correctly and generated the expected columns.")

        # Set the input dimension for the MLP model in args
        args.INPUT_DIM = len(feature_cols)

        # Validate that all labels in the dataset are covered by the global map
        for label_str in full_traffic_labels_list:
            if label_str not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_str}' not found in all_class_labels_global_map.")
        for label_in_df in df['label'].unique():
            if label_in_df not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_in_df}' from CSV not found in all_class_labels_global_map.")

        args.NUM_BASE_CLASSES = len(base_traffic_labels_str)
        args.NUM_ALL_CLASSES = len(all_class_labels_global_map)

        base_class_global_indices_sorted = sorted([all_class_labels_global_map[l] for l in base_traffic_labels_str])

        # --- Dataset Splitting Logic (unchanged from original) ---
        min_samples_for_stratify_df = df['label'].value_counts().min() >= 2 and len(df['label'].unique()) > 1
        df_train_val_pool, df_test_full = train_test_split(
            df, test_size=0.3, stratify=df['label'] if min_samples_for_stratify_df else None, random_state=42
        )

        df_base_pool = df_train_val_pool[df_train_val_pool['label'].isin(base_traffic_labels_str)].copy()
        df_new_pool = df_train_val_pool[df_train_val_pool['label'].isin(new_traffic_labels_str)].copy()

        df_train_decoop = pd.DataFrame(columns=df_base_pool.columns)

        for label_str in base_traffic_labels_str:
            class_df = df_base_pool[df_base_pool['label'] == label_str]

            if len(class_df) < num_samples_per_class:
                print(f"Warning: Not enough samples ({len(class_df)}) for class '{label_str}' to get {num_samples_per_class} for train. Taking all available.")
                train_samples = class_df.sample(frac=1, random_state=42)
            else:
                train_samples = class_df.sample(n=num_samples_per_class, random_state=42)

            df_train_decoop = pd.concat([df_train_decoop, train_samples], ignore_index=True)

        train_indices_used = df_train_decoop.index

        df_remaining_base_for_val = df_base_pool.drop(train_indices_used, errors='ignore')

        df_val = pd.concat([df_remaining_base_for_val, df_new_pool], ignore_index=True).sample(frac=1, random_state=42)

        df_test = df_test_full

        print(f"DECOOP Few-Shot Train (Base Classes Only): {len(df_train_decoop)} samples")
        print(f"Final Validation (Mixed Classes for Calibration): {len(df_val)} samples")
        print(f"Full Test (Mixed Classes): {len(df_test)} samples")
        print(f"Base classes for training: {base_traffic_labels_str}")
        if new_traffic_labels_str:
            print(f"New/OOD classes for evaluation: {new_traffic_labels_str}")

        print("\n--- Checking Class Distribution in Test Set (df_test) ---")

        test_class_counts_str = df_test['label'].value_counts().sort_index()
        print("Test set (df_test) class counts (string labels):")
        print(test_class_counts_str)

        print("\nTest set (df_test) class counts (global indices):")
        global_test_class_counts = {}
        for label_str, count in test_class_counts_str.items():
            global_idx = all_class_labels_global_map.get(label_str)
            if global_idx is not None:
                global_test_class_counts[global_idx] = count

        sorted_global_test_class_counts = sorted(global_test_class_counts.items())
        for global_idx, count in sorted_global_test_class_counts:
            local_base_idx_info = ''
            if global_idx in base_class_global_indices_sorted:
                local_base_idx_info = f" (Local Base Index {base_class_global_indices_sorted.index(global_idx)})"
            print(f"  Global Class {global_idx}{local_base_idx_info}: {count} samples")
        print("------------------------------------------------------")

        # New function to process DataFrame into numerical samples for MLP
        def process_dataframe_for_mlp(dataframe, feature_columns_for_extraction, all_class_labels_global_map_local, is_training_data_flag, base_class_global_indices_local=None, dataset_name_for_text=None): # 添加 dataset_name_for_text
            """
            Processes a pandas DataFrame to extract numerical features and labels,
            suitable for MLP training, and includes raw_text for LLM ZS classifier.
            """
            processed_list = []

            global_to_local_base_map = None
            if is_training_data_flag:
                if base_class_global_indices_local is None or not all_class_labels_global_map_local:
                    raise ValueError("For training data, base_class_global_indices and all_class_labels_global_map must be provided.")
                sorted_base_class_global_indices = sorted(list(set(base_class_global_indices_local)))
                if not sorted_base_class_global_indices:
                    raise ValueError("For training data, 'base_class_global_indices' cannot be empty.")
                global_to_local_base_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_base_class_global_indices)}

            for idx, row in dataframe.iterrows():
                global_label_str = row['label']
                global_label_numerical = all_class_labels_global_map_local.get(global_label_str)

                if global_label_numerical is None:
                    raise ValueError(f"Label '{global_label_str}' not found in all_class_labels_global_map for row index {idx}.")

                # Extract numerical features directly from the row using the specified columns
                numerical_features = torch.tensor([row[col] for col in feature_columns_for_extraction], dtype=torch.float32)

                target_label_numerical = global_label_numerical
                if is_training_data_flag:
                    # Map global label to local base class index for training
                    target_label_numerical = global_to_local_base_map.get(global_label_numerical, -1)
                    if target_label_numerical == -1:
                        # This should ideally not happen if `df_train_decoop` is correctly filtered
                        # to only include base classes relevant for a specific detector's ID training.
                        raise ValueError(f"Training data error: label {global_label_str} ({global_label_numerical}) not in base_class set for training (idx={idx}). This indicates an issue in `df_train_decoop`'s content or `base_class_global_indices_local` setup.")

                if not (isinstance(target_label_numerical, int) and target_label_numerical >= 0):
                    raise ValueError(f"Label mapping error at idx={idx}: mapped label ({target_label_numerical}) is not a valid non-negative integer. Original label: {global_label_str} ({global_label_numerical})")

                # 为 LLM ZS 分类器生成原始文本
                full_input_text = convert_feature_to_prompt_text(dataset_name_for_text, row) # 使用完整的 row 来生成文本

                processed_list.append({
                    "features": numerical_features,
                    "labels": torch.tensor(target_label_numerical, dtype=torch.long), # Local base class index or global index
                    "global_labels": torch.tensor(global_label_numerical, dtype=torch.long), # Always global index
                    "raw_text": full_input_text # 新增原始文本字段
                })
            return processed_list


        train_numerical_list = process_dataframe_for_mlp(df_train_decoop, feature_cols, all_class_labels_global_map, is_training_data_flag=True, base_class_global_indices_local=base_class_global_indices_sorted, dataset_name_for_text=args.dataset_name)
        val_numerical_list = process_dataframe_for_mlp(df_val, feature_cols, all_class_labels_global_map, is_training_data_flag=False, dataset_name_for_text=args.dataset_name)
        test_numerical_list = process_dataframe_for_mlp(df_test, feature_cols, all_class_labels_global_map, is_training_data_flag=False, dataset_name_for_text=args.dataset_name)

        train_dataset = MLPCSVTrafficDataset(train_numerical_list)
        val_dataset = MLPCSVTrafficDataset(val_numerical_list)
        test_dataset = MLPCSVTrafficDataset(test_numerical_list)

        # Save processed numerical data to cache
        torch.save({
            "train_numerical_list": train_numerical_list,
            "val_numerical_list": val_numerical_list,
            "test_numerical_list": test_numerical_list,
            "args": args, # Save updated args (now contains INPUT_DIM, LLM_MODEL_NAME, LABEL_TOKEN_MAP)
            "base_class_indices": base_class_global_indices_sorted,
            "ood_labels_excluded": ood_labels_to_exclude
        }, cache_file)
    return train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted
