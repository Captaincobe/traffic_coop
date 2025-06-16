import os
import pandas as pd
from scipy.io import arff # Assuming this is from SciPy
import numpy as np # For np.nan, though pd.NA is also an option
data_name = "ISCXVPN2016"
dataset_path = f"/home/icdm/code/trafficCOOP/datasets/{data_name}/"
folder_path = os.path.join(dataset_path, "Scenario_B/")
output_csv = os.path.join(dataset_path, f"{data_name}.csv")

# Define the master list of all columns in the desired final order.
# This should be the union of all possible attributes from your different ARFF file schemas.
# Based on our discussion (Schema A & Schema B combined):
master_column_list = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'std_fiat', 'std_biat',
    'flowPktsPerSecond', 'flowBytesPerSecond', 'min_flowiat', 'max_flowiat',
    'mean_flowiat', 'std_flowiat', 'min_active', 'mean_active', 'max_active',
    'std_active', 'min_idle', 'mean_idle', 'max_idle', 'std_idle', 'class1'
] # Total 26 columns

dataframes = []

# Recursively traverse subdirectories
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".arff"):
            file_path = os.path.join(root, filename)
            print(f"正在读取: {file_path}")
            
            try:
                # data is a NumPy structured array, meta contains header info
                data, meta = arff.loadarff(file_path)
                
                # Convert NumPy structured array to pandas DataFrame
                # Column names are taken from the 'meta' object's attribute names
                # which are also the dtype names of the structured array 'data'
                df = pd.DataFrame(data)

                # Decode byte strings in object columns (often nominal attributes)
                for col in df.select_dtypes([object]):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else (x.strip() if isinstance(x, str) else x) )
                
                # Ensure numeric columns are properly numeric, coercing errors if necessary
                # This is important if '?' for missing values were not converted to NaN by loadarff
                # or if numbers are strings.
                # For simplicity, we assume loadarff and DataFrame conversion handle types well,
                # but explicit conversion might be needed for specific problematic columns.
                # For example, if '?' are present as strings:
                # for col_name in meta.names(): # Iterate through original attribute names
                #     if meta[col_name][0].lower() == 'numeric' or meta[col_name][0].lower() == 'real' or meta[col_name][0].lower() == 'integer':
                #         if col_name in df.columns:
                #            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')


                dataframes.append(df)

            except arff.ArffError as e: # Specific error from scipy.io.arff
                print(f"读取 ARFF 文件失败 (ArffError): {file_path}, 错误: {e}")
                print("请检查该ARFF文件的头部（@RELATION, @ATTRIBUTE）和数据行格式是否符合标准。")
            except Exception as e:
                print(f"读取失败: {file_path}, 错误: {e}")

# Merge all DataFrames
if dataframes:
    # Concatenate DataFrames. This will naturally create a union of columns.
    # Columns present in some DataFrames but not others will be filled with NaN.
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

    # Ensure all columns from master_column_list exist in merged_df.
    # Add any missing columns from the master_list (these will be filled with NaN/NA).
    # Then, reorder/select columns according to master_column_list.
    # This also drops any columns in merged_df that are not in master_column_list.
    merged_df = merged_df.reindex(columns=master_column_list)
    merged_df.fillna(0, inplace=True)
    # Optional: If you want to replace all NaN with '?' for ARFF-like missing representation in CSV
    # merged_df.fillna('?', inplace=True)
    merged_df.rename(columns={'class1': 'label'}, inplace=True)
    # --- 新增：过滤掉“近乎空”或“零活动”的流量 ---
    initial_row_count = len(merged_df)
    
    # 定义判断“零活动”流的关键列
    # 这些列通常表示流量的核心活动：持续时间、每秒包数、每秒字节数
    activity_cols = ["duration", "flowPktsPerSecond", "flowBytesPerSecond"]
    
    # 过滤条件：保留那些在 activity_cols 中至少有一列不为 0.0 的行
    # df[activity_cols] == 0.0：检查这些列是否都为 0.0
    # .all(axis=1)：判断每行中所有指定列是否都为 0.0
    # ~：取反，保留那些不全为 0.0 的行
    df_filtered = merged_df[~((merged_df[activity_cols] == 0.0).all(axis=1))].copy()
    
    rows_removed = initial_row_count - len(df_filtered)
    if rows_removed > 0:
        print(f"Removed {rows_removed} 'near-empty' or 'zero-activity' flows from the dataset based on activity columns being zero.")
    else:
        print("No 'near-empty' flows found based on the defined criterion.")

    merged_df = df_filtered # 后续所有处理都使用过滤后的 DataFrame
    # ---------------------------------------------
    merged_df.to_csv(output_csv, index=False)
    print(f"已保存合并文件为：{output_csv}")
else:
    print("未找到任何 .arff 文件，或所有文件读取均失败")