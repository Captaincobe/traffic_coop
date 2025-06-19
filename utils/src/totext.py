import os
import pandas as pd
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import time

from sentence_transformers import SentenceTransformer
from datasetsM.preprocess import Z_Scaler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

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

def convert_feature_to_prompt_text(dataset_name, row):
    """
    Converts a traffic feature row into a domain-tokenized prompt string for LLMs.
    Adds explicit domain tags for protocol, service, flow duration, average packet size, and state.
    Uses categorical thresholds for duration (30s) and mean packet size (1000 bytes).
    Remaining features are appended as key=value for completeness.
    """
    # Common protocol mappings
    proto_map = {
        6: "TCP",
        17: "UDP",
        1: "ICMP",
        "tcp": "TCP",
        "udp": "UDP",
        "icmp": "ICMP"
    }
    # Common service/port mappings (destination port)
    service_map = {
        80: "HTTP",
        443: "HTTPS",
        21: "FTP",
        22: "SSH",
        23: "TELNET",
        25: "SMTP",
        53: "DNS",
        110: "POP3",
        143: "IMAP",
        993: "IMAPS",
        995: "POP3S",
        1935: "RTMP",
        554: "RTSP",
        5060: "SIP",
        1723: "PPTP",
        3306: "MySQL",
        3389: "RDP",
        8080: "HTTP-ALT"
    }
    # Protocol extraction & mapping
    proto = row.get("proto", row.get("Protocol", row.get("ProtocolName", "")))
    proto_str = proto_map.get(proto, str(proto))
    # Destination port extraction
    dst_port = row.get("dst_port", row.get("dport", row.get("Destination Port", "")))
    service_str = service_map.get(dst_port, str(dst_port))
    # Duration (categorical band)
    duration = row.get("duration", row.get("Flow Duration", 0.0))
    try:
        duration_val = float(duration)
    except Exception:
        duration_val = 0.0
    dur_band = "long-flow" if duration_val > 30.0 else "short-flow"
    # Mean packet size or flow bytes/s (categorical band)
    size_mean = row.get("mean_fiat",
                  row.get("Packet Length Mean",
                  row.get("Fwd Packet Length Mean",
                  row.get("Flow Bytes/s", 0.0))))
    try:
        size_val = float(size_mean)
    except Exception:
        size_val = 0.0
    size_desc = "large-pkt" if size_val > 1000 else "small-pkt"
    # State
    state = row.get('conn_state', row.get('Flow State', 'UNK'))
    # Compose prompt text
    text = (
        f"[PROTO] {proto_str} [SERVICE] {service_str} "
        f"[DURATION] {duration_val:.1f}s ({dur_band}) "
        f"[MEAN_PKT] {size_val:.0f} ({size_desc}) "
        f"[STATE] {state}"
    )
    # Append all other features as key=value (excluding those already used)
    exclude_keys = {
        'proto', 'dst_port', 'dport', 'Protocol', 'ProtocolName', 'Destination Port',
        'duration', 'Flow Duration',
        'mean_fiat', 'Packet Length Mean', 'Fwd Packet Length Mean', 'Flow Bytes/s',
        'conn_state', 'Flow State'
    }
    extras = " ".join([f"{k}={v}" for k, v in row.items() if k not in exclude_keys])
    return text + " " + extras


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

    # Unified cache file – contains fused features for the entire dataset (no OOD suffix needed)
    fused_cache_file_name = "fused_features_for_mlp_ALL.pt"
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

        # Log sample if explanation seems empty (summary+evidence)
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
        # fused_vector = torch.cat([textual_embedding,
        #                           numerical_features_tensor,
        #                           explanation_tensor,
        #                           p_logits_trans], dim=0)

        # --- Save both fused and disentangled feature components so that downstream
        #     ablation experiments can select arbitrary subsets without regenerating
        #     the cache. ---
        fused_data_list.append({
            # "fused_features": fused_vector,
            # component tensors (for fine‑grained ablation)
            "textual_emb": textual_embedding,                # (H,)
            "numerical_feats": numerical_features_tensor,    # (N_num,)
            "explanation_emb": explanation_tensor,           # (384,)
            # meta‑data / labels
            "global_labels": torch.tensor(global_label_numerical, dtype=torch.long),
            "raw_text": full_input_text,
            "prior_logits": p_g_logits                         # (C_all,)
        })

    torch.save(fused_data_list, fused_cache_file)
    print(f"Pre-fused features saved to: {fused_cache_file}")
    print(f"Total fused samples: {len(fused_data_list)}")

    return fused_data_list
