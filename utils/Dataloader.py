import math
import torch
from torch.utils.data import Dataset

# def convert_feature_to_prompt_text(feature_row) -> str:
#     prompt_prefix = "Traffic classification sample: "
#     feature_tokens = ", ".join([f"{col}: {val}" for col, val in feature_row.items()])
#     return prompt_prefix + feature_tokens

def get_formatted_feature_value(feature_name, value, format_str="", unit=""):
    if value == -1.0:
        # 对于间隔时间相关的特征
        if "fiat" in feature_name or "biat" in feature_name or "flowiat" in feature_name:
            return "undefined due to insufficient packets"
        # 对于活跃/空闲时间相关的特征
        if "active" in feature_name or "idle" in feature_name:
            return "not recorded due to flow characteristics"
        # 其他 -1.0
        return "not applicable"
    return f"{value:{format_str}}{unit}"

def convert_feature_to_prompt_text(dataset_name, feature_row) -> str:
    if dataset_name == "ISCXVPN2016":
        try:
            duration = float(feature_row.get("duration", 0))
            pkts_per_sec = float(feature_row.get("flowPktsPerSecond", 0))
            bytes_per_sec = float(feature_row.get("flowBytesPerSecond", 0))
            mean_fiat = float(feature_row.get("mean_fiat", 0))
            min_fiat = float(feature_row.get("min_fiat", 0))
            mean_biat = float(feature_row.get("mean_biat", 0))
            min_biat = float(feature_row.get("min_biat", 0))
            mean_active = float(feature_row.get("mean_active", 0))
            mean_idle = float(feature_row.get("mean_idle", 0))
        except Exception as e:
            raise ValueError(f"Feature row parsing error: {e}")

        prompt = (
            f"This network flow lasts for {duration:.1f} seconds. "
            f"It shows an average of {get_formatted_feature_value('flowPktsPerSecond', pkts_per_sec, '.2f', ' packets per second')} and {get_formatted_feature_value('flowBytesPerSecond', bytes_per_sec, '.2f', ' bytes per second')}. "
            f"The mean forward inter-arrival time range from {get_formatted_feature_value('min_fiat', min_fiat, '.3f', 's')} to {get_formatted_feature_value('mean_fiat', mean_fiat, '.3f', 's')} on average."
            f"The mean backward inter-arrival time range from {get_formatted_feature_value('min_biat', min_biat, '.3f', 's')} to {get_formatted_feature_value('mean_biat', mean_biat, '.3f', 's')} on average."
            f"It includes active periods averaging {mean_active:.3f}s and idle periods around {mean_idle:.3f}s."
        )
            # f"This network flow lasts for {duration:.1f} seconds. "
            # f"It shows an average of {pkts_per_sec:.2f} packets per second and {bytes_per_sec:.2f} bytes per second. "
            # f"The mean forward inter-arrival time range from {min_fiat:.3f}s to {mean_fiat:.3f}s on average."
            # f"The mean backward inter-arrival times range from {min_biat:.3f}s to {mean_biat:.3f}s."
    return prompt



# class LLMCSVTrafficDataset(Dataset):
#     # (Using the implementation from the previous response)
#     def __init__(self, dataset_name, dataframe, tokenizer, max_len, label_map, feature_columns, 
#                  label_column='label', is_training_data=False, 
#                  base_class_global_indices=None, all_class_labels_global_map=None):
#         self.dataframe = dataframe.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.label_map = label_map
#         self.feature_columns = feature_columns
#         self.label_column = label_column
#         self.is_training_data = is_training_data
#         self.dataset_name = dataset_name
#         self.prompt_prefix = "Traffic classification sample:"
#         self.max_seq_len = max_len  # 这样 __getitem__ 中就可以用

#         if self.is_training_data:
#             if base_class_global_indices is None:
#                 raise ValueError(
#                     "For training data, 'base_class_global_indices' must be provided (not None)."
#                 )
#             if self.label_map is None or not self.label_map: # 检查 self.label_map
#                 raise ValueError(
#                     "For training data, 'label_map' (all_class_labels_global_map) must be provided and non-empty."
#                 )
            
#             # 确保 base_class_global_indices 是排序好的，以保证本地索引的确定性
#             sorted_base_class_global_indices = sorted(list(set(base_class_global_indices)))
#             if not sorted_base_class_global_indices: # 如果基类列表为空
#                     raise ValueError("For training data, 'base_class_global_indices' cannot be empty.")

#             self.global_to_local_base_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_base_class_global_indices)}
#             # 验证所有 base_class_global_indices 是否真的存在于 self.label_map 的值中
#             # (这一步通常在创建 base_class_global_indices_sorted 时已经隐含处理了，但可以加一道保险)
#             for global_idx in sorted_base_class_global_indices:
#                 if global_idx not in self.label_map.values(): # 或者更准确地，检查它们是否是预期的基类标签对应的索引
#                     raise ValueError(f"Global base class index {global_idx} not found in the values of the provided label_map.")

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         global_label_str = row[self.label_column]
#         global_label_numerical = self.label_map.get(global_label_str)

#         if global_label_numerical is None:
#             raise ValueError(f"Label '{global_label_str}' not found in label_map.")

#         feature_text = convert_feature_to_prompt_text(self.dataset_name, row[self.feature_columns])
#         full_input = f"{self.prompt_prefix} {feature_text}"

#         tokenized = self.tokenizer(
#             full_input,
#             truncation=True,
#             max_length=self.max_seq_len,
#             padding="max_length",
#             return_tensors="pt"
#         )

#         if self.is_training_data:
#             # 映射为局部索引标签（用于 base class）
#             target_label_numerical = self.global_to_local_base_map.get(global_label_numerical, -1)
#             # print(f"[DEBUG] idx={idx}, global_label_str={global_label_str}, global_label_numerical={global_label_numerical}, mapped_local_label={target_label_numerical}")
#             if target_label_numerical == -1:
#                 raise ValueError(f"Training data error: label {global_label_str} ({global_label_numerical}) not in base_class set.")
#         else:
#             target_label_numerical = global_label_numerical

#         # Validation check: ensure label is in valid range
#         if not (isinstance(target_label_numerical, int) and target_label_numerical >= 0):
#             raise ValueError(f"Label mapping error at idx={idx}: mapped label ({target_label_numerical}) is not a valid non-negative integer. Original label: {global_label_str} ({global_label_numerical})")

#         return {
#             "input_ids": tokenized["input_ids"].squeeze(0),
#             "attention_mask": tokenized["attention_mask"].squeeze(0),
#             "labels": torch.tensor(target_label_numerical, dtype=torch.long),  # local label
#             "global_labels": torch.tensor(global_label_numerical, dtype=torch.long),
#             "raw_text": full_input # 
#         }
    
class LLMCSVTrafficDataset(Dataset):
    # 直接接收预分词后的数据列表
    def __init__(self, tokenized_data_list):
        self.tokenized_data_list = tokenized_data_list

    def __len__(self):
        return len(self.tokenized_data_list)

    def __getitem__(self, idx):
        return self.tokenized_data_list[idx]
