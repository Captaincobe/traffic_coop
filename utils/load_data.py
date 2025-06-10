import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from utils.Dataloader import LLMCSVTrafficDataset




# --- 1. Data Loading and Preprocessing (Using implementations from previous responses) ---
def load_and_prepare_data_for_llm(args, base_traffic_labels_str, all_class_labels_global_map):
    max_seq_len_for_llm = args.MAX_SEQ_LENGTH
    dataset_name = args.dataset_name
    # (Using the implementation from the previous response, ensure it returns base_class_global_indices_sorted)
    csv_path = f'/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/{dataset_name}--1.csv'
    cache_file = f"/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/tokenized.pt"
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
        for label_str in base_traffic_labels_str:
            if label_str not in all_class_labels_global_map: raise ValueError(f"Base class '{label_str}' not in map.")
        for label_in_df in df['label'].unique():
            if label_in_df not in all_class_labels_global_map: raise ValueError(f"Label '{label_in_df}' from CSV not in map.")

        args.NUM_BASE_CLASSES = len(base_traffic_labels_str)
        args.NUM_ALL_CLASSES = len(all_class_labels_global_map)
        base_class_global_indices_sorted = sorted([all_class_labels_global_map[l] for l in base_traffic_labels_str])
        # global label → local label
        df_base_only = df[df['label'].isin(base_traffic_labels_str)].copy()
        df_new_only = df[~df['label'].isin(base_traffic_labels_str)].copy()
        if len(df_base_only) == 0: raise ValueError("No base class data.")

        min_samples_base_strat = df_base_only['label'].value_counts().min() >=2 and len(df_base_only['label'].unique()) > 1
        df_train_decoop, df_base_remaining = train_test_split(df_base_only, test_size=0.3, stratify=df_base_only['label'] if min_samples_base_strat else None, random_state=42) \
            if len(df_base_only) >= 2 else (df_base_only, pd.DataFrame(columns=df.columns))
        
        # mixed data
        df_mixed_pool = pd.concat([df_base_remaining, df_new_only], ignore_index=True)
        if len(df_mixed_pool) == 0:
            print("Warning: Mixed pool empty. Using DECOOP train data for val/test.")
            min_samples_train_strat = df_train_decoop['label'].value_counts().min() >=2 and len(df_train_decoop['label'].unique()) > 1
            df_val, df_test = train_test_split(df_train_decoop, test_size=0.5, stratify=df_train_decoop['label'] if min_samples_train_strat else None, random_state=42) \
                if len(df_train_decoop) >=2 else (df_train_decoop, df_train_decoop)
        else:
            min_samples_mixed_strat = df_mixed_pool['label'].value_counts().min() >=2 and len(df_mixed_pool['label'].unique()) > 1
            df_val, df_test = train_test_split(df_mixed_pool, test_size=0.5, stratify=df_mixed_pool['label'] if min_samples_mixed_strat else None, random_state=42) \
                if len(df_mixed_pool) >=2 else (df_mixed_pool, df_mixed_pool)

        print(f"DECOOP Train: {len(df_train_decoop)}, val: {len(df_val)}, Test: {len(df_test)}")

        train_dataset = LLMCSVTrafficDataset(dataset_name, df_train_decoop, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=True, base_class_global_indices=base_class_global_indices_sorted)
        val_dataset = LLMCSVTrafficDataset(dataset_name, df_val, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=False)
        test_dataset = LLMCSVTrafficDataset(dataset_name, df_test, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=False)
        torch.save({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "args": args,
            "base_class_indices": base_class_global_indices_sorted,
        }, cache_file)
    return train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted

