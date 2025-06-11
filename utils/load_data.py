import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from utils.Dataloader import LLMCSVTrafficDataset, convert_feature_to_prompt_text


# --- 1. Data Loading and Preprocessing (Using implementations from previous responses) ---
def load_data(args, full_traffic_labels_list, all_class_labels_global_map, ood_labels_to_exclude=None):
    max_seq_len_for_llm = args.MAX_SEQ_LENGTH
    dataset_name = args.dataset_name

    if ood_labels_to_exclude is None:
        ood_labels_to_exclude = []

    base_traffic_labels_str = [label for label in full_traffic_labels_list if label not in ood_labels_to_exclude]
    new_traffic_labels_str = [label for label in full_traffic_labels_list if label in ood_labels_to_exclude]

    csv_path = f'/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/{dataset_name}.csv'
    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"
    cache_file = f"/home/icdm/code/trafficCOOP/datasets/{dataset_name}/raw/tokenized_{ood_suffix}.pt"

    if os.path.exists(cache_file):
        print(f"Loading tokenized datasets from cache: {cache_file}")
        cached = torch.load(cache_file)

        train_dataset = LLMCSVTrafficDataset(cached["train_tokenized_list"])
        val_dataset = LLMCSVTrafficDataset(cached["val_tokenized_list"])
        test_dataset = LLMCSVTrafficDataset(cached["test_tokenized_list"])
        args.NUM_BASE_CLASSES = cached["args"].NUM_BASE_CLASSES
        args.NUM_ALL_CLASSES = cached["args"].NUM_ALL_CLASSES
        base_class_global_indices_sorted = cached["base_class_indices"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.LLM_MODEL_NAME)

        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col != 'label']
        if not feature_cols: raise ValueError("No feature columns found.")

        for label_str in base_traffic_labels_str + new_traffic_labels_str:
            if label_str not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_str}' not found in all_class_labels_global_map.")
        for label_in_df in df['label'].unique():
            if label_in_df not in all_class_labels_global_map:
                raise ValueError(f"Label '{label_in_df}' from CSV not found in all_class_labels_global_map.")

        args.NUM_BASE_CLASSES = len(base_traffic_labels_str)
        args.NUM_ALL_CLASSES = len(all_class_labels_global_map)

        base_class_global_indices_sorted = sorted([all_class_labels_global_map[l] for l in base_traffic_labels_str])
        
        df_base_only = df[df['label'].isin(base_traffic_labels_str)].copy()
        df_new_only = df[df['label'].isin(new_traffic_labels_str)].copy()

        if len(df_base_only) == 0:
            raise ValueError("No base class data available after excluding OOD labels. Adjust your 'ood_labels_to_exclude' or dataset.")

        min_samples_base_strat = df_base_only['label'].value_counts().min() >=2 and len(df_base_only['label'].unique()) > 1
        df_train_decoop, df_base_remaining = train_test_split(df_base_only, test_size=0.3, stratify=df_base_only['label'] if min_samples_base_strat else None, random_state=42) \
            if len(df_base_only) >= 2 else (df_base_only, pd.DataFrame(columns=df.columns))
        
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

        print(f"DECOOP Train (Base Classes Only): {len(df_train_decoop)} samples")
        print(f"Validation (Mixed Classes): {len(df_val)} samples")
        print(f"Test (Mixed Classes): {len(df_test)} samples")
        print(f"Base classes for training: {base_traffic_labels_str}")
        if new_traffic_labels_str:
            print(f"New/OOD classes for evaluation: {new_traffic_labels_str}")
        # --- 检查 df_val 中的类别分布 ---
        print("\n--- Checking Class Distribution in Validation Set (df_val) ---")
        
        # 打印字符串标签的计数
        val_class_counts_str = df_val['label'].value_counts().sort_index()
        print("Validation set (df_val) class counts (string labels):")
        print(val_class_counts_str)

        # 打印全局索引的计数，方便与日志中的 'class X' 对应
        print("\nValidation set (df_val) class counts (global indices):")
        global_val_class_counts = {}
        for label_str, count in val_class_counts_str.items():
            global_idx = all_class_labels_global_map.get(label_str)
            if global_idx is not None:
                global_val_class_counts[global_idx] = count
        
        sorted_global_val_class_counts = sorted(global_val_class_counts.items())
        for global_idx, count in sorted_global_val_class_counts:
            # 尝试获取本地基类索引，如果它是一个基类的话
            local_base_idx_info = ''
            if global_idx in base_class_global_indices_sorted:
                local_base_idx_info = f" (Local Base Index {base_class_global_indices_sorted.index(global_idx)})"
            print(f"  Global Class {global_idx}{local_base_idx_info}: {count} samples")
        print("------------------------------------------------------")

        # --- 进行分词操作，将结果保存为列表 ---
        def tokenize_dataframe(dataframe, tokenizer, max_seq_len, all_class_labels_global_map, feature_columns, is_training_data, dataset_name, base_class_global_indices=None):
            tokenized_list = []
            
            global_to_local_base_map = None
            if is_training_data:
                if base_class_global_indices is None or not all_class_labels_global_map:
                    raise ValueError("For training data, base_class_global_indices and all_class_labels_global_map must be provided.")
                sorted_base_class_global_indices = sorted(list(set(base_class_global_indices)))
                if not sorted_base_class_global_indices:
                    raise ValueError("For training data, 'base_class_global_indices' cannot be empty.")
                global_to_local_base_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_base_class_global_indices)}


            for idx, row in dataframe.iterrows():
                global_label_str = row['label']
                global_label_numerical = all_class_labels_global_map.get(global_label_str)

                if global_label_numerical is None:
                    raise ValueError(f"Label '{global_label_str}' not found in all_class_labels_global_map for row index {idx}.")


                feature_text = convert_feature_to_prompt_text(dataset_name, row)

                prompt_prefix = "Traffic classification sample:"
                full_input = f"{prompt_prefix} {feature_text}"

                tokenized = tokenizer(
                    full_input,
                    truncation=True,
                    max_length=max_seq_len,
                    padding="max_length",
                    return_tensors="pt"
                )

                target_label_numerical = global_label_numerical
                if is_training_data:
                    target_label_numerical = global_to_local_base_map.get(global_label_numerical, -1)
                    if target_label_numerical == -1:
                        raise ValueError(f"Training data error: label {global_label_str} ({global_label_numerical}) not in base_class set for training (idx={idx}). This should not happen if df_train_decoop is correctly filtered.")

                if not (isinstance(target_label_numerical, int) and target_label_numerical >= 0):
                    raise ValueError(f"Label mapping error at idx={idx}: mapped label ({target_label_numerical}) is not a valid non-negative integer. Original label: {global_label_str} ({global_label_numerical})")


                tokenized_list.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "labels": torch.tensor(target_label_numerical, dtype=torch.long),
                    "global_labels": torch.tensor(global_label_numerical, dtype=torch.long),
                    "raw_text": full_input
                })
            return tokenized_list


        train_tokenized_list = tokenize_dataframe(df_train_decoop, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=True, dataset_name=args.dataset_name, base_class_global_indices=base_class_global_indices_sorted)
        val_tokenized_list = tokenize_dataframe(df_val, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=False, dataset_name=args.dataset_name)
        test_tokenized_list = tokenize_dataframe(df_test, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, feature_cols, is_training_data=False, dataset_name=args.dataset_name)


        train_dataset = LLMCSVTrafficDataset(train_tokenized_list)
        val_dataset = LLMCSVTrafficDataset(val_tokenized_list)
        test_dataset = LLMCSVTrafficDataset(test_tokenized_list)
        

        torch.save({
            "train_tokenized_list": train_tokenized_list,
            "val_tokenized_list": val_tokenized_list,
            "test_tokenized_list": test_tokenized_list,
            "args": args,
            "base_class_indices": base_class_global_indices_sorted,
            "ood_labels_excluded": ood_labels_to_exclude
        }, cache_file)
    return train_dataset, val_dataset, test_dataset, args, base_class_global_indices_sorted


