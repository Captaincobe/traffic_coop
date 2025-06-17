# utils_mlp/load_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Import the refactored Dataset class
from utils.Dataloader import MLPCSVTrafficDataset

# Helper function to get the list of features that are expected to be standardized
# and used as input to the MLP. This function replicates the relevant part
# from `datasetsM/preprocess.py` to ensure consistency in feature selection.
# In a larger project, this would ideally be imported from a shared config module.
# NOTE: This is now only for defining the numerical part if you were to fuse them.
# The `_get_features_to_standardize_for_loading` will still be used to determine the *numerical* input part.
# But for loading the *fused* data, we won't use it directly to select columns.
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

def loadData(args, full_traffic_labels_list, all_class_labels_global_map, ood_labels_to_exclude=None, prefused_data_list=None):
    """
    Loads, splits, and processes traffic data for MLP training and evaluation.
    Handles caching of processed numerical data.
    
    Args:
        args: An argparse.Namespace object containing configuration parameters
              like `dataset_name`, `SAMPLES_PER_CLASS`, `DEVICE`, etc.
        full_traffic_labels_list: A list of all possible traffic class labels (strings).
        all_class_labels_global_map: A dictionary mapping string labels to their global integer IDs.
        ood_labels_to_exclude: A list of string labels to be considered OOD for the current run.
        prefused_data_list (list): A list of dictionaries containing pre-fused 'fused_features'
                                   and 'global_labels' as generated by the `fuse_features_and_cache` function.
                                   If None, it tries to load from cache or raise an error if not found.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, updated_args, base_class_global_indices_sorted)
    """
    dataset_name = args.dataset_name
    num_samples_per_class = args.SAMPLES_PER_CLASS

    if ood_labels_to_exclude is None:
        ood_labels_to_exclude = []

    base_traffic_labels_str = [label for label in full_traffic_labels_list if label not in ood_labels_to_exclude]
    new_traffic_labels_str = [label for label in full_traffic_labels_list if label in ood_labels_to_exclude]
    dataset_root='/home/icdm/code/trafficCOOP/datasetsM'

    # The cache file name now expects fused features
    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"
    cache_file_name = f"fused_features_for_mlp_{ood_suffix}_few_shot_{num_samples_per_class}.pt"
    cache_file = os.path.join(dataset_root, dataset_name, "raw", cache_file_name)

    # Use pre-fused data if provided, otherwise try to load from cache
    if prefused_data_list is not None:
        print(f"Using provided pre-fused data for MLP data loading.")
        all_processed_data = prefused_data_list
    elif os.path.exists(cache_file):
        print(f"Loading pre-fused data from cache: {cache_file}")
        all_processed_data = torch.load(cache_file)
    else:
        raise FileNotFoundError(f"Pre-fused data cache not found at {cache_file}. Please run the feature fusion step first, or provide `prefused_data_list`.")

    # Determine input_dim from the first sample's fused_features
    if not all_processed_data:
        raise ValueError("No data found in pre-fused list. Cannot determine INPUT_DIM.")
    args.INPUT_DIM = all_processed_data[0]["fused_features"].shape[0]
    print(f"MLP Input Dimension (Fused Features): {args.INPUT_DIM}")

    args.NUM_BASE_CLASSES = len(base_traffic_labels_str)
    args.NUM_ALL_CLASSES = len(all_class_labels_global_map)

    base_class_global_indices_sorted = sorted([all_class_labels_global_map[l] for l in base_traffic_labels_str])
    
    # Reconstruct a DataFrame-like structure for splitting, using global_labels
    temp_df = pd.DataFrame([
        {
            'fused_features': sample['fused_features'],
            'label': sample['global_labels'].item(),
            # Keep raw_text so later Dataset conversion can access it
            'raw_text': sample.get('raw_text', "")
        }
        for sample in all_processed_data
    ])
    label_counts = temp_df['label'].value_counts().sort_index()
    print(label_counts)

    min_samples_for_stratify_df = temp_df['label'].value_counts().min() >= 2 and len(temp_df['label'].unique()) > 1
    df_train_val_pool, df_test_full = train_test_split(
        temp_df, test_size=0.3, stratify=temp_df['label'] if min_samples_for_stratify_df else None, random_state=42
    )

    df_base_pool = df_train_val_pool[df_train_val_pool['label'].isin(base_class_global_indices_sorted)].copy()
    
    # New classes for validation/test are those global labels that are NOT base classes
    new_class_global_indices = [all_class_labels_global_map[l] for l in new_traffic_labels_str]
    df_new_pool = df_train_val_pool[df_train_val_pool['label'].isin(new_class_global_indices)].copy()

    df_train_decoop = pd.DataFrame(columns=df_base_pool.columns)

    for global_label_idx in base_class_global_indices_sorted:
        class_df = df_base_pool[df_base_pool['label'] == global_label_idx]
        
        if len(class_df) < num_samples_per_class:
            print(f"Warning: Not enough samples ({len(class_df)}) for global class {global_label_idx} to get {num_samples_per_class} for train. Taking all available.")
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
    
    # Convert dataframes back to list of dicts for MLPCSVTrafficDataset
    def df_to_mlp_dataset_format(dataframe, is_training_data_flag, base_class_global_indices_local=None):
        processed_list = []
        global_to_local_base_map = None
        if is_training_data_flag:
            sorted_base_class_global_indices = sorted(list(set(base_class_global_indices_local)))
            global_to_local_base_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_base_class_global_indices)}

        for idx, row in dataframe.iterrows():
            fused_features = row['fused_features']
            global_label_numerical = row['label']

            target_label_numerical = global_label_numerical
            if is_training_data_flag:
                target_label_numerical = global_to_local_base_map.get(global_label_numerical, -1)
                if target_label_numerical == -1:
                    raise ValueError(f"Training data error: label {global_label_numerical} not in base_class set for training (idx={idx}).")

            processed_list.append({
                "features": fused_features, # Now contains fused features
                "labels": torch.tensor(target_label_numerical, dtype=torch.long), # Local base class index or global index
                "raw_text": row['raw_text'],
                "global_labels": torch.tensor(global_label_numerical, dtype=torch.long), # Always global index
            })
        return processed_list

    train_numerical_list = df_to_mlp_dataset_format(df_train_decoop, is_training_data_flag=True, base_class_global_indices_local=base_class_global_indices_sorted)
    val_numerical_list = df_to_mlp_dataset_format(df_val, is_training_data_flag=False)
    test_numerical_list = df_to_mlp_dataset_format(df_test, is_training_data_flag=False)

    train_dataset = MLPCSVTrafficDataset(train_numerical_list)
    val_dataset = MLPCSVTrafficDataset(val_numerical_list)
    test_dataset = MLPCSVTrafficDataset(test_numerical_list)
    
    # The cache for fused data is handled by the `fuse_features_and_cache` function.
    # We do not re-save here to avoid redundant or inconsistent caching.
    
    return train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted