# utils/load_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from utils.Dataloader import LLMCSVTrafficDataset


def load_data_leave(args, full_traffic_labels_list, all_class_labels_global_map, ood_labels_to_exclude=None):
    """
    Loads and prepares data for LLM training, incorporating Leave-One-Out logic.

    Args:
        args: Argparse object containing configuration.
        full_traffic_labels_list (list): A list of all possible traffic labels (strings).
        all_class_labels_global_map (dict): A mapping from string labels to global numerical indices.
        ood_labels_to_exclude (list, optional): A list of string labels that should be treated as OOD (new classes).
                                                 If None or empty, all classes in full_traffic_labels_list
                                                 will be considered base classes. Defaults to None.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted)
    """
    max_seq_len_for_llm = args.MAX_SEQ_LENGTH
    dataset_name = args.dataset_name

    # Determine base and new traffic labels based on ood_labels_to_exclude
    if ood_labels_to_exclude is None:
        ood_labels_to_exclude = []

    base_traffic_labels_str = [label for label in full_traffic_labels_list if label not in ood_labels_to_exclude]
    new_traffic_labels_str = [label for label in full_traffic_labels_list if label in ood_labels_to_exclude]

    csv_path = f'/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/{dataset_name}--1.csv'
    # Modify cache file name to reflect OOD configuration
    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"
    cache_file = f"/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/tokenized_{ood_suffix}.pt"

    if os.path.exists(cache_file):
        print(f"Loading tokenized datasets from cache: {cache_file}")
        cached = torch.load(cache_file)
        train_dataset = cached["train"]
        val_dataset = cached["val"]
        test_dataset = cached["test"]
        args.NUM_BASE_CLASSES = cached["args"].NUM_BASE_CLASSES
        args.NUM_ALL_CLASSES = cached["args"].NUM_ALL_CLASSES
        base_class_global_indices_sorted = cached["base_class_indices"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.LLM_MODEL_NAME)

        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col != 'label']
        if not feature_cols: raise ValueError("No feature columns found.")

        # Validate that all base and new labels are in the global map
        for label_str in base_traffic_labels_str + new_traffic_labels_str:
            if label_str not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_str}' not found in all_class_labels_global_map.")
        # Validate that all labels in the DataFrame are in the global map
        for label_in_df in df['label'].unique():
            if label_in_df not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_in_df}' from CSV not found in all_class_labels_global_map.")

        args.NUM_BASE_CLASSES = len(base_traffic_labels_str)
        args.NUM_ALL_CLASSES = len(all_class_labels_global_map) # This remains the total number of unique classes

        base_class_global_indices_sorted = sorted([all_class_labels_global_map[l] for l in base_traffic_labels_str])
        
        # Split data: DECOOP train uses only base classes.
        # Val and Test sets can contain both base and new classes.
        df_base_only = df[df['label'].isin(base_traffic_labels_str)].copy()
        df_new_only = df[df['label'].isin(new_traffic_labels_str)].copy()

        if len(df_base_only) == 0:
            raise ValueError("No base class data available after excluding OOD labels. Adjust your 'ood_labels_to_exclude' or dataset.")

        # For DECOOP training (PromptLearner, Subclassifiers), we use only base classes.
        min_samples_base_strat = df_base_only['label'].value_counts().min() >= 2 and len(df_base_only['label'].unique()) > 1
        df_train_decoop, df_base_remaining = train_test_split(
            df_base_only, 
            test_size=0.3, # This split ratio can be adjusted
            stratify=df_base_only['label'] if min_samples_base_strat else None, 
            random_state=42
        ) if len(df_base_only) >= 2 else (df_base_only, pd.DataFrame(columns=df.columns))
        
        # Validation and test sets will contain a mix of base and new classes
        df_mixed_pool = pd.concat([df_base_remaining, df_new_only], ignore_index=True)

        if len(df_mixed_pool) == 0:
            print("Warning: Mixed pool (for validation/test) is empty. Using DECOOP train data for val/test, which might not accurately reflect OOD performance.")
            # Fallback to splitting df_train_decoop if mixed pool is empty
            min_samples_train_strat = df_train_decoop['label'].value_counts().min() >= 2 and len(df_train_decoop['label'].unique()) > 1
            df_val, df_test = train_test_split(
                df_train_decoop, 
                test_size=0.5, # Split train data into val and test
                stratify=df_train_decoop['label'] if min_samples_train_strat else None, 
                random_state=42
            ) if len(df_train_decoop) >= 2 else (df_train_decoop, df_train_decoop)
        else:
            min_samples_mixed_strat = df_mixed_pool['label'].value_counts().min() >= 2 and len(df_mixed_pool['label'].unique()) > 1
            df_val, df_test = train_test_split(
                df_mixed_pool, 
                test_size=0.5, # Split mixed pool into val and test
                stratify=df_mixed_pool['label'] if min_samples_mixed_strat else None, 
                random_state=42
            ) if len(df_mixed_pool) >= 2 else (df_mixed_pool, df_mixed_pool)

        print(f"DECOOP Train (Base Classes Only): {len(df_train_decoop)} samples")
        print(f"Validation (Mixed Classes): {len(df_val)} samples")
        print(f"Test (Mixed Classes): {len(df_test)} samples")
        print(f"Base classes for training: {base_traffic_labels_str}")
        if new_traffic_labels_str:
            print(f"New/OOD classes for evaluation: {new_traffic_labels_str}")


        train_dataset = LLMCSVTrafficDataset(dataset_name, df_train_decoop, tokenizer, max_seq_len_for_llm, 
                                             all_class_labels_global_map, feature_cols, 
                                             is_training_data=True, base_class_global_indices=base_class_global_indices_sorted)
        # For validation and test, `is_training_data` is False because they might contain OOD samples
        val_dataset = LLMCSVTrafficDataset(dataset_name, df_val, tokenizer, max_seq_len_for_llm, 
                                           all_class_labels_global_map, feature_cols, 
                                           is_training_data=False)
        test_dataset = LLMCSVTrafficDataset(dataset_name, df_test, tokenizer, max_seq_len_for_llm, 
                                            all_class_labels_global_map, feature_cols, 
                                            is_training_data=False)
        
        torch.save({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "args": args,
            "base_class_indices": base_class_global_indices_sorted,
            "ood_labels_excluded": ood_labels_to_exclude # Save which labels were excluded for caching
        }, cache_file)
    return train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted