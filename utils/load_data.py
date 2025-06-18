# utils_mlp/load_data.py

# utils_mlp/load_data.py
import string
from transformers import StoppingCriteria, StoppingCriteriaList
import os
import json
import re
import time
# import openai   # assumes OPENAI_API_KEY is in env
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import SentenceTransformer
from datasetsM.preprocess import Z_Scaler
from utils.Dataloader import convert_feature_to_prompt_text
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel


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
            'raw_text': sample.get('raw_text', ""),
            'p_g_logits': sample['p_g_logits']
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
                "features": fused_features,                                # Fused vector incl. explanation
                "labels": torch.tensor(target_label_numerical, dtype=torch.long),
                "raw_text": row['raw_text'],
                "global_labels": torch.tensor(global_label_numerical, dtype=torch.long),
                "p_g_logits": row['p_g_logits']                            # tensor with softlogits
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
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -1] == stop_id for stop_id in self.stop_ids)

def chat(messages, model, tokenizer, max_tokens=512):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample=False
    # --- Ensure generation config has no sampling params when do_sample=False ----
    if not do_sample:
        # Override potentially inherited defaults that trigger warnings
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)]),
        do_sample=do_sample
    )
    # Only add sampling-specific kwargs when sampling is enabled
    if do_sample:
        gen_kwargs.update(dict(top_p=0.95, temperature=0.7))

    output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response
# --- NEW: Pre-fuse textual and numerical features using LLM embeddings ---
# This function is defined inline for demonstration but could be a separate utility.
def fuse_features_and_cache(args, all_labels_list, all_class_labels_global_map, ood_labels_to_exclude, dataset_name):
    # Get paths
    dataset_root = '/home/icdm/code/trafficCOOP/datasetsM'
    csv_path_raw = os.path.join(dataset_root, dataset_name, 'raw', f"{dataset_name}.csv")  # Original raw CSV

    ood_suffix = "_".join(sorted(ood_labels_to_exclude)) if ood_labels_to_exclude else "all_base"
    fused_cache_file_name = f"fused_features_for_mlp_{ood_suffix}_few_shot_{args.SAMPLES_PER_CLASS}.pt"
    fused_cache_file = os.path.join(dataset_root, dataset_name, "raw", fused_cache_file_name)

    if os.path.exists(fused_cache_file):
        print(f"Loading pre-fused features from cache: {fused_cache_file}")
        return torch.load(fused_cache_file)

    print("\n--- Starting Pre-fusion of Textual and Numerical Features ---")

    df_raw = pd.read_csv(csv_path_raw)
    # ---- Load LLM tokenizer & model ONCE (lazy global) ----
    if "llm_tokenizer" not in globals():
        print("[LLM] Loading tokenizer & model for textual CLS embeddings …")
        global llm_tokenizer, llm_model, llm_embedding_dim
        llm_tokenizer = AutoTokenizer.from_pretrained(args.LLM_MODEL_NAME)
        llm_model = AutoModel.from_pretrained(args.LLM_MODEL_NAME).to(args.DEVICE)
        llm_model.eval() # Set LLM to evaluation mode, as we're only extracting features
        llm_embedding_dim = llm_model.config.hidden_size
    else:
        llm_embedding_dim = llm_model.config.hidden_size
    # ---- Load local chat model ONCE (lazy global, no pipeline) ----
    if "chat_tokenizer" not in globals():
        print("[GEN] Loading local chat model (no pipeline) for JSON explanation …")
        local_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Qwen/Qwen2.5-7B-Instruct  deepseek-ai/deepseek-coder-6.7b-instruct
        global chat_tokenizer, chat_model
        chat_tokenizer = AutoTokenizer.from_pretrained(local_model_name, trust_remote_code=True)
        chat_tokenizer.padding_side = "left"           # faster for causal LM
        chat_model = AutoModelForCausalLM.from_pretrained(
            local_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(args.DEVICE)
        chat_model.eval()
    numerical_feature_cols = _get_features_to_standardize_for_loading(dataset_name)

    # Fit numerical scaler on the full raw dataset
    numerical_scaler = Z_Scaler()
    numerical_scaler.fit_transform(df_raw[numerical_feature_cols].copy())

    fused_data_list = []

    numerical_output_dim = len(numerical_feature_cols)
    exp_embedding_dim = 384                                # MiniLM sentence‑BERT size
    p_logits_dim = len(all_class_labels_global_map)        # one per global class
    print(f"LLM embedding dimension: {llm_embedding_dim}")
    print(f"Numerical features dimension: {numerical_output_dim}")
    print(f"Explanation embedding dimension: {exp_embedding_dim}")
    print(f"Prior logits dimension: {p_logits_dim}")
    total_fused_dim = llm_embedding_dim + numerical_output_dim + exp_embedding_dim + p_logits_dim
    print(f"Total fused feature dimension for MLP: {total_fused_dim}")

    # ---- Load lightweight encoder for explanation text ----
    exp_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Candidate label list must match prompt
    candidate_labels = [
        "VPN-MAIL", "VPN-STREAMING", "VPN-VOIP", "VPN-BROWSING",
        "CHAT", "STREAMING", "MAIL", "P2P", "BROWSING", "VOIP"
    ]

    for idx, row in df_raw.iterrows():
        global_label_str = row['label']
        global_label_numerical = all_class_labels_global_map.get(global_label_str)
        if global_label_numerical is None:
            continue  # Skip if label not in map

        # 2. Get Scaled Numerical Features
        numerical_features_scaled = numerical_scaler.fit_transform(
            row[numerical_feature_cols].values.reshape(1, -1)
        ).squeeze(0)
        numerical_features_tensor = torch.tensor(numerical_features_scaled.astype(float), dtype=torch.float32)

        # ---- Build human‑readable metrics block for the prompt (no raw JSON) ----
        # Remove label to avoid leakage
        row_prompt = row.copy()
        if "label" in row_prompt:
            row_prompt = row_prompt.drop(labels=["label"])

        # Build bullet lines for the key numerical features (rounded) in the preferred order
        def _fmt(val):
            """
            Format helper:
            - Any negative numeric value (int or float) is treated as missing → 'miss'
            - Floats are rounded to 4 significant digits.
            - Non‑empty strings are returned verbatim; empty strings/None → 'miss'
            """
            if isinstance(val, (int, float)) and val < 0:
                return "missing"
            if isinstance(val, float):
                return f"{val:.4g}"
            return val if val not in ("", None) else "missing"

        feature_order = [
            "duration", "total_fiat", "total_biat",
            "min_fiat", "min_biat", "max_fiat", "max_biat",
            "mean_fiat", "mean_biat", "std_fiat", "std_biat",
            "flowPktsPerSecond", "flowBytesPerSecond",
            "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
            "min_active", "mean_active", "max_active", "std_active",
            "min_idle", "mean_idle", "max_idle", "std_idle"
        ]
        feature_lines = [
            f"- {fname}: {_fmt(row_prompt.get(fname, '?'))}"
            for fname in feature_order
        ]
        feature_block = "\n".join(feature_lines)

        # 3. Build prompt

        prompt = f"""
            You are an expert in encrypted‑traffic analysis.  
            Your task is to assign the most appropriate category (label) to the flow below.

            Flow metrics:{feature_block}
            Candidate labels: {candidate_labels}

            Return **EXACTLY** the following 5 lines, DO NOT output anthing other.
            Evidences lines must reference only the metric names above:

            SUMMARY: <a clear, one-sentence, and concise summary of traffic>
            EVIDENCE1: <field>=<value>
            EVIDENCE2: <field>=<value>
            EVIDENCE3: <field>=<value>
            PRED: <label from {candidate_labels}>,<confidence 1-10>
        """.strip()

        # --- Generate explanation via unified chat / fallback call ---
        start = time.time()
        if hasattr(chat_tokenizer, "chat_template") and chat_tokenizer.chat_template:
            gen_out = chat(
                [
                    {
                        "role": "system",
                        "content": ("You are an expert in encrypted-traffic analysis." "Always answer concisely and back every classification with clear evidence.")
                    },
                    {"role": "user", "content": f"{prompt}"}
                ],
                chat_model,
                chat_tokenizer
            )
        else:
            inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
            output_ids = chat_model.generate(**inputs, max_new_tokens=256)
            gen_out = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"=== Explanation generation time: {time.time() - start:.2f} s ===")

        # ---- Extract the block that starts with SUMMARY and ends with the PRED line ----
        # This matches everything from "SUMMARY:" up to and including the first "PRED:" line.
        m_block = re.search(
            r"(SUMMARY\s*:.*?PRED\s*:[^\n\r]*$)", gen_out, re.S | re.I)
        if m_block:
            explanation_text = m_block.group(1).strip()
        else:
            # Fallback: use full output if pattern not found
            explanation_text = gen_out.strip()

        # Parse PRED line to build p_g_logits (label + confidence)
        pred_label_str, conf = "BROWSING", 5
        m_pred = re.search(
            r"PRED\s*:\s*([A-Za-z0-9\-]+)\s*,\s*([1-9]|10)",
            explanation_text,
            re.I
        )
        if m_pred:
            pred_label_str = m_pred.group(1).upper().strip()
            conf = int(m_pred.group(2))

        # Log sample if explanation seems empty
        if explanation_text.count("SUMMARY") == 0:
            print(f"[WARN] No proper OUTPUT block for sample {idx}; using raw LLM output.")

        print(f"{idx} gen_out: {gen_out}")
        print(f"{idx} explanation_text: {explanation_text[:120]}...")
        # Use explanation_text for embedding
        with torch.no_grad():
            exp_emb = exp_encoder.encode(explanation_text, batch_size=16, show_progress_bar=False)  # (384,)
        explanation_tensor = torch.tensor(exp_emb, dtype=torch.float32)

        # --- p_g_logits vector ---
        p_g_logits = torch.zeros(len(all_class_labels_global_map), dtype=torch.float32)
        if pred_label_str in all_class_labels_global_map:
            p_idx = all_class_labels_global_map[pred_label_str]
            # Simple linear mapping: confidence 1‑10 -> 0.1‑1.0
            p_g_logits[p_idx] = conf / 10.0
        eps = 1e-4
        p_logits_trans = torch.logit(torch.clamp(p_g_logits, eps, 1 - eps))

        # 1. Get Textual Embedding from LLM
        feature_text = convert_feature_to_prompt_text(dataset_name, row)
        full_input_text = f"Traffic classification sample: {feature_text}"

        tokenized_input = llm_tokenizer(
            full_input_text,
            truncation=True,
            max_length=args.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(args.DEVICE)

        with torch.no_grad():
            outputs = llm_model(**tokenized_input)
            textual_embedding = outputs.last_hidden_state[:, 0].squeeze(0).cpu()  # (H,)

        # 4. Concatenate (Fuse) them
        fused_vector = torch.cat([textual_embedding,
                                  numerical_features_tensor,
                                  explanation_tensor,
                                  p_logits_trans], dim=0)

        fused_data_list.append({
            "fused_features": fused_vector,
            "global_labels": torch.tensor(global_label_numerical, dtype=torch.long),
            "raw_text": full_input_text,
            "p_g_logits": p_g_logits
        })

    torch.save(fused_data_list, fused_cache_file)
    print(f"Pre-fused features saved to: {fused_cache_file}")
    print(f"Total fused samples: {len(fused_data_list)}")

    return fused_data_list
