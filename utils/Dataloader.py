import math
import torch
from torch.utils.data import Dataset

def convert_feature_to_prompt_text(feature_row) -> str:
    prompt_prefix = "Traffic classification sample: "
    feature_tokens = ", ".join([f"{col}: {val}" for col, val in feature_row.items()])
    return prompt_prefix + feature_tokens

def traffic_features_to_text(dataset_name, row):
    if dataset_name == "ISCXVPN2016":
        result = {
            "duration": round(row["duration"], 4),
            "min_fiat": round(row["min_fiat"], 4),
            "min_biat": round(row["min_biat"], 4),
            "max_fiat": round(row["max_fiat"], 4),
            "max_biat": round(row["max_biat"], 4),
            "mean_fiat": round(row["mean_fiat"], 4),
            "mean_biat": round(row["mean_biat"], 4),

            "flow_pkts_per_sec": round(row["flowPktsPerSecond"], 4),
            "flow_bytes_per_sec": round(row["flowBytesPerSecond"], 4),

            "min_flowiat": round(row["min_flowiat"], 4),
            "max_flowiat": round(row["max_flowiat"], 4),
            "mean_flowiat": round(row["mean_flowiat"], 4),
            "std_flowiat": round(row["std_flowiat"], 4),

            "min_active": round(row["min_active"], 4),
            "active_mean": round(row["mean_active"], 4),
            "max_active": round(row["max_active"], 4),
            "std_active": round(row["std_active"], 4),

            "min_idle": round(row["min_idle"], 4),
            "idle_mean": round(row["mean_idle"], 4),
            "max_idle": round(row["max_idle"], 4),
            "std_idle": round(row["std_idle"], 4),
        }

        if not math.isnan(row["total_fiat"]):
            result["total_fiat"] = round(row["total_fiat"], 4)
            result["total_biat"] = round(row["total_biat"], 4)
        else:
            result["std_fiat"] = round(row["std_fiat"], 4)
            result["std_biat"] = round(row["std_biat"], 4)

        formatted_string_parts = []
        for key, value in result.items():
            formatted_string_parts.append(f"{key}: {value}")
        
        finall = " ; ".join(formatted_string_parts)

        return finall


class LLMCSVTrafficDataset(Dataset):
    # (Using the implementation from the previous response)
    def __init__(self, dataset_name, dataframe, tokenizer, max_len, label_map, feature_columns, 
                 label_column='label', is_training_data=False, 
                 base_class_global_indices=None, all_class_labels_global_map=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = label_map
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.is_training_data = is_training_data
        self.dataset_name = dataset_name

        if self.is_training_data:
            if base_class_global_indices is None:
                raise ValueError(
                    "For training data, 'base_class_global_indices' must be provided (not None)."
                )
            if self.label_map is None or not self.label_map: # 检查 self.label_map
                raise ValueError(
                    "For training data, 'label_map' (all_class_labels_global_map) must be provided and non-empty."
                )
            
            # 确保 base_class_global_indices 是排序好的，以保证本地索引的确定性
            sorted_base_class_global_indices = sorted(list(set(base_class_global_indices)))
            if not sorted_base_class_global_indices: # 如果基类列表为空
                    raise ValueError("For training data, 'base_class_global_indices' cannot be empty.")

            self.global_to_local_base_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_base_class_global_indices)}
            # 验证所有 base_class_global_indices 是否真的存在于 self.label_map 的值中
            # (这一步通常在创建 base_class_global_indices_sorted 时已经隐含处理了，但可以加一道保险)
            # for global_idx in sorted_base_class_global_indices:
            #     if global_idx not in self.label_map.values(): # 或者更准确地，检查它们是否是预期的基类标签对应的索引
            #         raise ValueError(f"Global base class index {global_idx} not found in the values of the provided label_map.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # feature_text = convert_feature_to_prompt_text(self.df.iloc[idx][self.feature_cols])
        # full_input = self.prompt_prefix + " " + feature_text
        # tokenized = self.tokenizer(full_input, truncation=True, max_length=self.max_seq_len, padding="max_length", return_tensors="pt")
        
        text_sequence = traffic_features_to_text(self.dataset_name, row[self.feature_columns] )
        global_label_numerical = self.label_map.get(row[self.label_column])
        if global_label_numerical is None:
            raise ValueError(f"Label '{row[self.label_column]}' not found in label_map. Available: {list(self.label_map.keys())}")

        encoding = self.tokenizer.encode_plus(
            text_sequence, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=True, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')
        target_label_numerical = -1
        if self.is_training_data:
            target_label_numerical = self.global_to_local_base_map.get(global_label_numerical)
            if target_label_numerical is None:
                raise ValueError(f"Training data error: Global label {global_label_numerical} ('{row[self.label_column]}') not base class.")
        else: 
            target_label_numerical = global_label_numerical
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding and encoding['token_type_ids'] is not None else torch.tensor([]),
            'labels': torch.tensor(target_label_numerical, dtype=torch.long),
            'global_labels': torch.tensor(global_label_numerical, dtype=torch.long) # Always include global label for potential use
        }
