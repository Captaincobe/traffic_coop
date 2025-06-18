"""
Generate English instruction-tuning data (train/val JSONL) for MLM Zero-Shot fine-tuning.

Usage:
    python utils_mlp/generate_jsonl_en.py \
        --fused_path path/to/data.pt \
        --shots_per_class 25 \
        --val_ratio 0.2 \
        --out_dir dataset/ISCXVPN2016
"""
import json, random, argparse, os, torch
from collections import defaultdict

# Mapping label → verbalizer  (make sure single-token)
LABEL_TOKEN_MAP = {
    "VPN-MAIL": "vpnmail",
    "VPN-STREAMING": "vpnstream",
    "VPN-VOIP": "vpnvoip",
    "VPN-BROWSING": "vpnbrowse",
    "CHAT": "chat",
    "STREAMING": "streaming",
    "MAIL": "mail",
    "FT": "file transfer",
    "VPN-FT": "vpnfile",
    "P2P": "p2p",
    "BROWSING": "browse",
    "VOIP": "voip",
    "VPN-P2P": "vpnp2p",
    "VPN-CHAT": "vpnchat"
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused_path", default="datasetsM/ISCXVPN2016/raw/fused_features_for_mlp_VOIP_few_shot_200.pt")
    ap.add_argument("--shots_per_class", type=int, default=200)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--out_dir", default="datasetsM/ISCXVPN2016")
    return ap.parse_args()

def make_prompt_en(feature_text: str) -> str:
    """
    English instruction-style prompt ending with [MASK] token.
    """
    return (
        "[INSTRUCTION] You are a network-traffic analyst. "
        "Based on the following flow description, classify the traffic category.\n"
        f"{feature_text}\n"
        "[QUESTION] What is the traffic category?\n"
        "[ANSWER] [MASK]"
    )

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fused_list = torch.load(args.fused_path)
    buckets = defaultdict(list)          # global_label_id → list of feature_text
    id2label = {}                        # reverse lookup (for readability)

    # Each dict has 'raw_text'  (assured by previous edits)
    for sample in fused_list:
        gid = int(sample["global_labels"])
        txt = sample.get("raw_text", "")
        buckets[gid].append(txt)

    print(f"Found {len(buckets)} distinct global labels.")

    data_all = []
    random.seed(42)

    for gid, txt_list in buckets.items():
        # map gid back to label string
        label_str = next(k for k,v in all_class_labels_global_map.items() if v == gid)
        verbal = LABEL_TOKEN_MAP[label_str]
        chosen = random.sample(txt_list, min(args.shots_per_class, len(txt_list)))
        for ft in chosen:
            data_all.append({
                "text" : make_prompt_en(ft),
                "label": verbal
            })

    random.shuffle(data_all)
    n_val = int(len(data_all)*args.val_ratio)
    val, train = data_all[:n_val], data_all[n_val:]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path   = os.path.join(args.out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf8") as fw:
        for d in train:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf8") as fw:
        for d in val:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(val)} val samples to '{args.out_dir}'")

if __name__ == "__main__":
    # Import the global label map from your main script (adjust import path as needed)
    from main import all_class_labels_global_map
    main()