from torch.utils.data import Dataset


# --- Domain-aware prompt textification for traffic features ---
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


class MLPCSVTrafficDataset(Dataset):
    """
    A PyTorch Dataset for traffic data, designed to load pre-processed
    numerical features for MLP models.
    """
    def __init__(self, data_list):
        """
        Initializes the dataset with a list of pre-processed data samples.
        Each sample in `data_list` should be a dictionary containing:
        - 'features': a torch.Tensor of numerical feature vectors
        - 'labels': a torch.Tensor (long) for local (base) class index
        - 'global_labels': a torch.Tensor (long) for global class index
        - 'raw_text': (Optional) A string representation of features for LLM input. # 新增注释，明确支持 raw_text
        """
        self.data_list = data_list

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        """
        return self.data_list[idx]