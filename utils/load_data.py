# utils/load_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from utils.Dataloader import LLMCSVTrafficDataset, convert_feature_to_prompt_text



# --- 1. Data Loading and Preprocessing (Using implementations from previous responses) ---
def loadData(args, full_traffic_labels_list, all_class_labels_global_map, ood_labels_to_exclude=None):
    max_seq_len_for_llm = args.MAX_SEQ_LENGTH
    dataset_name = args.dataset_name
    num_samples_per_class = args.SAMPLES_PER_CLASS # 获取新参数

    if ood_labels_to_exclude is None:
        ood_labels_to_exclude = []

    base_traffic_labels_str = [label for label in full_traffic_labels_list if label not in ood_labels_to_exclude]
    new_traffic_labels_str = [label for label in full_traffic_labels_list if label in ood_labels_to_exclude]
    dataset_root='/home/icdm/code/trafficCOOP/datasets'
    csv_path = os.path.join(dataset_root, dataset_name, "raw", f"{dataset_name}.csv")

    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"

    cache_file_name = f"tokenized_{ood_suffix}_few_shot_{num_samples_per_class}.pt"
    cache_file = os.path.join(dataset_root, dataset_name, "raw", cache_file_name)

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
        
        # --- 针对最终实验设置的数据集划分逻辑 ---
        # 1. 首先，将原始数据划分为一个用于训练/验证的池子（train_val_pool）和一个完整的测试集（df_test_full）
        # 保持 train_val_pool 和 test_full 之间的分层抽样
        min_samples_for_stratify_df = df['label'].value_counts().min() >= 2 and len(df['label'].unique()) > 1
        df_train_val_pool, df_test_full = train_test_split(
            df, test_size=0.3, stratify=df['label'] if min_samples_for_stratify_df else None, random_state=42
        )

        # 2. 从 train_val_pool 中分离出基类和新类
        df_base_pool = df_train_val_pool[df_train_val_pool['label'].isin(base_traffic_labels_str)].copy()
        df_new_pool = df_train_val_pool[df_train_val_pool['label'].isin(new_traffic_labels_str)].copy()

        # 3. 为训练集进行 Few-shot 采样（仅从基类池中）
        df_train_decoop = pd.DataFrame(columns=df_base_pool.columns) # few-shot train set

        for label_str in base_traffic_labels_str:
            class_df = df_base_pool[df_base_pool['label'] == label_str]
            
            if len(class_df) < num_samples_per_class:
                print(f"Warning: Not enough samples ({len(class_df)}) for class '{label_str}' to get {num_samples_per_class} for train. Taking all available.")
                train_samples = class_df.sample(frac=1, random_state=42) # Take all available for train
            else:
                train_samples = class_df.sample(n=num_samples_per_class, random_state=42)
            
            df_train_decoop = pd.concat([df_train_decoop, train_samples], ignore_index=True)


        # 4. 构建最终验证集 (df_val): 包含除了 few-shot 训练样本外，所有剩余的基类样本和所有新类样本
        # 获取用于 few-shot 训练的样本的原始索引
        train_indices_used = df_train_decoop.index

        # 从基类池中剔除已用于训练的样本
        df_remaining_base_for_val = df_base_pool.drop(train_indices_used, errors='ignore') 
        
        # 验证集是剩余的基类样本加上所有新类样本，并打乱
        df_val = pd.concat([df_remaining_base_for_val, df_new_pool], ignore_index=True).sample(frac=1, random_state=42) 

        # 5. 将完整的测试集赋值给 df_test
        df_test = df_test_full

        print(f"DECOOP Few-Shot Train (Base Classes Only): {len(df_train_decoop)} samples")
        print(f"Final Validation (Mixed Classes for Calibration): {len(df_val)} samples")
        print(f"Full Test (Mixed Classes): {len(df_test)} samples")
        print(f"Base classes for training: {base_traffic_labels_str}")
        if new_traffic_labels_str:
            print(f"New/OOD classes for evaluation: {new_traffic_labels_str}")
        
        # --- 检查 df_test 中的类别分布 ---
        print("\n--- Checking Class Distribution in Test Set (df_test) ---")
        
        # 打印字符串标签的计数
        test_class_counts_str = df_test['label'].value_counts().sort_index()
        print("Test set (df_test) class counts (string labels):")
        print(test_class_counts_str)

        # 打印全局索引的计数，方便与日志中的 'class X' 对应
        print("\nTest set (df_test) class counts (global indices):")
        global_test_class_counts = {}
        for label_str, count in test_class_counts_str.items():
            global_idx = all_class_labels_global_map.get(label_str)
            if global_idx is not None:
                global_test_class_counts[global_idx] = count
        
        sorted_global_test_class_counts = sorted(global_test_class_counts.items())
        for global_idx, count in sorted_global_test_class_counts:
            # 尝试获取本地基类索引，如果它是一个基类的话
            local_base_idx_info = ''
            if global_idx in base_class_global_indices_sorted:
                local_base_idx_info = f" (Local Base Index {base_class_global_indices_sorted.index(global_idx)})"
            print(f"  Global Class {global_idx}{local_base_idx_info}: {count} samples")
        print("------------------------------------------------------")

        def tokenize_dataframe(dataframe, tokenizer, max_seq_len, all_class_labels_global_map, is_training_data, dataset_name, base_class_global_indices=None):
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


        train_tokenized_list = tokenize_dataframe(df_train_decoop, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, is_training_data=True, dataset_name=args.dataset_name, base_class_global_indices=base_class_global_indices_sorted)
        val_tokenized_list = tokenize_dataframe(df_val, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, is_training_data=False, dataset_name=args.dataset_name)
        test_tokenized_list = tokenize_dataframe(df_test, tokenizer, max_seq_len_for_llm, all_class_labels_global_map, is_training_data=False, dataset_name=args.dataset_name)


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
