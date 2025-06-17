import os
import random
import argparse
from typing import final

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# ===============================
# Argument Parsing
# ===============================
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='CICIDS',
                        choices=["TONIoT", "DoHBrw", "CICIDS", "CICMalMen"],
                        help='which dataset to use')
    parser.add_argument('--num_all', type=int, default=1000000, choices=[2000, 5000])
    parser.add_argument('--b', dest='b', action='store_true', default=False,
                        help='True if you want binary classification.')
    parser.add_argument('--n_small', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=9999)
    return parser.parse_args()


# ===============================
# Z-Score Normalizer
# ===============================
class Z_Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
        if torch.is_tensor(data):
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
            std = torch.from_numpy(self.std).type_as(data).to(data.device)
        else:
            mean = self.mean
            std = self.std
        return (data - mean) / (std + 1e-6)


# ===============================
# Dataset Configuration
# ===============================
def get_dataset_config(name):
    # CSV_FILES: final = ['dataset.csv']
    if name == 'DoHBrw':
        CSV_FILES: final = ['l2-benign.csv', 'l2-malicious.csv']
        FEATURES: final = [
        'src_ip',
        'dst_ip',
        'src_port',
        'dst_port',
        'Timestamp',
        'Duration',
        'FlowBytesSent',
        'FlowSentRate',
        'FlowBytesReceived',
        'FlowReceivedRate',
        'PacketLengthVariance',
        'PacketLengthStandardDeviation',
        'PacketLengthMean',
        'PacketLengthMedian',
        'PacketLengthMode',
        'PacketLengthSkewFromMedian',
        'PacketLengthSkewFromMode',
        'PacketLengthCoefficientofVariation',
        'PacketTimeVariance',
        'PacketTimeStandardDeviation',
        'PacketTimeMean',
        'PacketTimeMedian',
        'PacketTimeMode',
        'PacketTimeSkewFromMedian',
        'PacketTimeSkewFromMode',
        'PacketTimeCoefficientofVariation',
        'ResponseTimeTimeVariance',
        'ResponseTimeTimeStandardDeviation',
        'ResponseTimeTimeMean',
        'ResponseTimeTimeMedian',
        'ResponseTimeTimeMode',
        'ResponseTimeTimeSkewFromMedian',
        'ResponseTimeTimeSkewFromMode',
        'ResponseTimeTimeCoefficientofVariation',
        'label'
        ]
        FEATURES_TO_STANDARDIZE: final = [
        'Duration',
        'FlowBytesSent',
        'FlowSentRate',
        'FlowBytesReceived',
        'FlowReceivedRate',
        'PacketLengthVariance',
        'PacketLengthStandardDeviation',
        'PacketLengthMean',
        'PacketLengthMedian',
        'PacketLengthMode',
        'PacketLengthSkewFromMedian',
        'PacketLengthSkewFromMode',
        'PacketLengthCoefficientofVariation',
        'PacketTimeVariance',
        'PacketTimeStandardDeviation',
        'PacketTimeMean',
        'PacketTimeMedian',
        'PacketTimeMode',
        'PacketTimeSkewFromMedian',
        'PacketTimeSkewFromMode',
        'PacketTimeCoefficientofVariation',
        'ResponseTimeTimeVariance',
        'ResponseTimeTimeStandardDeviation',
        'ResponseTimeTimeMean',
        'ResponseTimeTimeMedian',
        'ResponseTimeTimeMode',
        'ResponseTimeTimeSkewFromMedian',
        'ResponseTimeTimeSkewFromMode',
        'ResponseTimeTimeCoefficientofVariation',
        ]
        MAPPING: final = {
        "Benign": 0,
        "Malicious": 1,
        }
    elif name == 'TONIoT':
        CSV_FILES: final = ['Network_dataset_' + str(i+1) + '.csv' for i in range(23)]
        # CSV_FILES: final = ['TONIoT-train.csv','TONIoT-val.csv','TONIoT-test.csv']
        FEATURES: final = [
            'Timestamp','src_ip','src_port','dst_ip','dst_port','proto','service', # 7
            'duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts', # 8
            'dst_ip_bytes',
            'dns_query',
            'dns_qclass',
            'dns_qtype',
            'dns_rcode',
            'dns_AA',
            'dns_RD',
            'dns_RA',
            'dns_rejected',
            'ssl_version',
            'ssl_cipher',
            'ssl_resumed',
            'ssl_established',
            'ssl_subject',
            'ssl_issuer',
            'http_trans_depth',
            'http_method',
            'http_uri',
            'http_referrer',
            'http_version',
            'http_request_body_len',
            'http_response_body_len',
            'http_status_code',
            'http_user_agent',
            'http_orig_mime_types',
            'http_resp_mime_types',
            'weird_name',
            'weird_addl',
            'weird_notice',
            'label',
            'type',
        ]
        FEATURES_TO_STANDARDIZE: final = [
            'proto',
            'service',
            'duration',
            'src_bytes',
            'dst_bytes',
            'conn_state',
            'missed_bytes',
            'src_pkts',
            'src_ip_bytes',
            'dst_pkts',
            'dst_ip_bytes',
            'dns_query',    
            'dns_rcode',
            'dns_AA',
            'dns_RD',
            'dns_RA',
            'dns_rejected',
            'ssl_version',
            'ssl_cipher',
            'ssl_resumed',
            'ssl_established',
            'ssl_subject',
            'ssl_issuer',
            'http_trans_depth',
            'http_method',
            'http_uri',
            'http_referrer',
            'http_version',
            'http_request_body_len',
            'http_response_body_len',
            'http_status_code',
            'http_user_agent',
            'http_orig_mime_types',
            'http_resp_mime_types',
            'weird_name',
            'weird_addl',
            'weird_notice',
            ]
        MAPPING: final = {
            "normal": 0,
            "backdoor": 1,
            "ddos": 2,
            "dos": 3,
            "injection": 4,
            "mitm": 5,
            "password": 6,
            "ransomware": 7,
            "scanning": 8,
            "xss": 9
        }
    elif name == 'CICIDS':
        CSV_FILES: final = ['dataset_1.csv', 'dataset_2.csv']
        FEATURES:final = [
            'Flow ID','src_ip','src_port','dst_ip','dst_port','Protocol','Timestamp', # 7
            'Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets', # 5
            'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std', # 4
            'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std', # 4
            'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min', # 6
            'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total', # 6
            'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags', # 6
            'Bwd URG Flags','Fwd Header Length','Bwd Header Length', 'Fwd Packets/s','Bwd Packets/s','Min Packet Length', # 6
            'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance',  # 4
            'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count', # 8
            'Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size', # 4
            'Fwd Header Length2', 'Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate',
            'Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward',
            'act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
            'label'
        ]
        FEATURES_TO_STANDARDIZE: final = [
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
            # remove ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
        ]
        MAPPING: final = {
        'BENIGN': 0, 
        'DoS GoldenEye': 1,
        'PortScan': 2,
        'DoS Slowhttptest': 3,
        'Web Attack  Brute Force': 4,
        'Bot': 5,
        'Web Attack  Sql Injection': 6,
        'Web Attack  XSS': 7,
        'Infiltration': 8,
        'DDoS': 9,
        'DoS slowloris': 10,
        'Heartbleed': 11,
        'FTP-Patator': 12,
        'DoS Hulk': 13,
        "SSH-Patator": 14,
        }
    elif name == 'CICAndMal2019':
        CSV_FILES: final = ['UNSW-NB15_1.csv','UNSW-NB15_2.csv','UNSW-NB15_3.csv','UNSW-NB15_4.csv']
        FEATURES: final = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'urg_count', 'rst_count', 
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
        'Radius', 'Covariance', 'Variance', 'Weight', 
        ]
        FEATURES_TO_STANDARDIZE: final = [

        ]
        MAPPING: final = {
        "Normal": 0,
        "Generic": 1,
        "Exploits": 2,
        "Fuzzers": 3,
        "DoS": 4,
        "Reconnaissance": 5,
        "Analysis": 6,
        "Backdoors": 7,
        "Shellcode": 8,
        "Worms": 9
    }

        CSV_FILES: final = ['UNSW-NB15_1.csv','UNSW-NB15_2.csv','UNSW-NB15_3.csv','UNSW-NB15_4.csv']
        FEATURES: final = [
        'srcip',
        'sport',
        'dstip',
        'dsport',
        'proto',
        'state',
        'dur',
        'sbytes',
        'dbytes',
        'sttl',
        'dttl',
        'sloss',
        'dloss',
        'service',
        'sload',
        'dload',
        'spkts',
        'dpkts',
        'swin',
        'dwin',
        'stcpb',
        'dtcpb',
        'smeansz',
        'dmeansz',
        'trans_depth',
        'res_bdy_len',
        'sjit',
        'djit',
        'stime',
        'ltime',
        'sintpkt',
        'dintpkt',
        'tcprtt',
        'synack',
        'ackdat',
        'is_sm_ips_ports',
        'ct_state_ttl',
        'ct_flw_http_mthd',
        'is_ftp_login',
        'ct_ftp_cmd',
        'ct_srv_src',
        'ct_srv_dst',
        'ct_dst_ltm',
        'ct_src_ltm',
        'ct_src_dport_ltm',
        'ct_dst_sport_ltm',
        'ct_dst_src_ltm',
        'attack_cat',
        'label'
        ]
        FEATURES_TO_STANDARDIZE: final = [
        'dur',
        'sbytes',
        'dbytes',
        'sttl',
        'dttl',
        'sloss',
        'dloss',
        'sload',
        'dload',
        'spkts',
        'dpkts',
        'swin',
        'dwin',
        'stcpb',
        'dtcpb',
        'smeansz',
        'dmeansz',
        'trans_depth',
        'res_bdy_len',
        'sjit',
        'djit',
        'sintpkt',
        'dintpkt',
        'tcprtt',
        'synack',
        'ackdat',
        'is_sm_ips_ports',
        'ct_state_ttl',
        'ct_flw_http_mthd',
        'is_ftp_login',
        'ct_ftp_cmd',
        'ct_srv_src',
        'ct_srv_dst',
        'ct_dst_ltm',
        'ct_src_ltm',
        'ct_src_dport_ltm',
        'ct_dst_sport_ltm',
        'ct_dst_src_ltm',
        ]
        MAPPING: final = {
        "Normal": 0,
        "Generic": 1,
        "Exploits": 2,
        "Fuzzers": 3,
        "DoS": 4,
        "Reconnaissance": 5,
        "Analysis": 6,
        "Backdoors": 7,
        "Shellcode": 8,
        "Worms": 9
    }
    else:
        raise ValueError("Unsupported dataset")
    return CSV_FILES, FEATURES, FEATURES_TO_STANDARDIZE, MAPPING


# ===============================
# Balanced Sampling
# ===============================
def create_balanced_sample(df, n_per_class, seed):
    sample_dfs = []
    unique_labels = df['label'].unique()
    total_needed = n_per_class * len(unique_labels)

    for label in unique_labels:
        class_df = df[df['label'] == label]
        sample_dfs.append(class_df.sample(min(n_per_class, len(class_df)), random_state=seed))

    remaining = total_needed - sum(len(s) for s in sample_dfs)
    if remaining > 0:
        remaining_df = df[~df.index.isin(pd.concat(sample_dfs).index)]
        sample_dfs.append(remaining_df.sample(remaining, random_state=seed))

    return pd.concat(sample_dfs)


# ===============================
# Dataset Processor
# ===============================
def process(df, dataset_name, features_to_std, mapping):
    if dataset_name == 'TONIoT':
        mask = df[features_to_std] == '-'
        for feat in features_to_std:
            df[feat] = LabelEncoder().fit_transform(df[feat].astype(str))
        df[mask] = 0
    elif dataset_name == 'CICIDS':
        drop_cols = ['Flow ID', 'Bwd PSH Flags', 'Bwd URG Flags']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    else:
        df['label'] = df['label'].str.strip()

    df['label'] = df['label'].map(mapping)

    # Drop rows where features are max or min
    max_vals = df[features_to_std].max()
    min_vals = df[features_to_std].min()
    df = df[~(df[features_to_std].isin(max_vals) | df[features_to_std].isin(min_vals)).any(axis=1)]

    # df[features_to_std] = Z_Scaler().fit_transform(df[features_to_std])
    return df

def remove_unnecessary_columns(df):
    unnecessary_cols = ['Timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port']
    existing_cols = [col for col in unnecessary_cols if col in df.columns]
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)
    return df

# ===============================
# Main Preprocessing Function
# ===============================
def pre_processing(dataset_name, num_all, seed, binary, n_small):
    files, features, features_to_std, mapping = get_dataset_config(dataset_name)
    dataset_path = os.path.join('../datasets', dataset_name, 'raw')
    df = pd.DataFrame(columns=features)

    print("loading dataset...")
    for f in files:
        path = os.path.join(dataset_path, f)
        df_local = pd.read_csv(path, header=None if dataset_name == 'UNSW15' else 0, low_memory=False).fillna(0)
        df_local.columns = features
        df = pd.concat([df, df_local], ignore_index=True)

    if dataset_name == 'TONIoT':
        df['label'] = df['type'].str.strip()
        df.drop(columns=['type'], inplace=True)

    if n_small > 0:
        df = create_balanced_sample(df, n_small, seed)

    df = process(df, dataset_name, features_to_std, mapping)
    # df = remove_unnecessary_columns(df)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[:min(num_all, len(df))]

    if binary:
        df['label'] = df['label'].apply(lambda x: 1 if x != 0 else 0)
        filename = f"{dataset_name}-{n_small}-binary.csv"
    else:
        filename = f"{dataset_name}-{n_small}.csv"

    df.to_csv(os.path.join(dataset_path, filename), index=False)
    print("Sample counts per class:")
    print(df['label'].value_counts())
    print("Saved to:", dataset_path)


# ===============================
# Entry Point
# ===============================
if __name__ == '__main__':
    args = parse_arguments()
    print(f"Dataset selected: {args.dataset_name}")
    pre_processing(args.dataset_name, args.num_all, args.seed, args.b, args.n_small)
    print("Done!")