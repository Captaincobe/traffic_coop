import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Configuration and Hyperparameters for Model Training")
    parser.add_argument('--num_base_classes', dest='NUM_BASE_CLASSES', type=int, default=-1)
    parser.add_argument('--num_all_classes', dest='NUM_ALL_CLASSES', type=int, default=-1)

    # Add LLM-specific parameters for MLMZeroShotClassifier
    parser.add_argument('--LLM_MODEL_NAME', type=str, default="bert-base-uncased", # Or "FacebookAI/roberta-base"
                        choices=["bert-base-uncased","FacebookAI/roberta-base"],
                        help='Name of the pre-trained LLM model to use for Zero-Shot Classification.')
    parser.add_argument('--max_seq_length', dest='MAX_SEQ_LENGTH', type=int, default=128,
                    help='Maximum sequence length for LLM input.')
    parser.add_argument('--label_token_map', dest='LABEL_TOKEN_MAP', default=
    {
        "VPN-MAIL": "vpnmail", "VPN-STREAMING": "vpnstream", "VPN-VOIP": "vpncall",
        "Browse":"web" ,"CHAT":"chat" ,"STREAMING":"stream" ,"MAIL":"mail",
        "FT":"ftp", "VPN-FT":"vpnftp", "VPN-P2P":"vpnshare", "VPN-Browse":"vpnweb",
        "VOIP":"call",  "P2P":"sharing", "VPN-CHAT":"vpnchat"
    }, help='Mapping from traffic labels to their verbalizer tokens for LLM zero-shot classification.')
    # 注意：这里的 LABEL_TOKEN_MAP 需要根据你的数据集和具体标签进行调整。


    parser.add_argument('--dataset_name', type=str, default='ISCXVPN2016',
                        choices=["ISCXVPN2016", "ISCXTor2016", "USTC-TFC2016", "CIC-Darknet2020"])
    # Few-shot training parameter
    parser.add_argument('--samples_per_class', dest='SAMPLES_PER_CLASS', type=int, default=200,
                        help='Number of samples per base class for few-shot training.')

    # Added MLP-specific parameters for defining network architecture.
    parser.add_argument('--input_dim', dest='INPUT_DIM', type=int, default=-1,
                        help='Dimension of input features for MLP. This will be set dynamically by load_data.py.')
    parser.add_argument('--mlp_hidden_dims_zs', dest='MLP_HIDDEN_DIMS_ZS', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions for the Zero-Shot MLP classifier (e.g., --mlp_hidden_dims_zs 128 64).')
    parser.add_argument('--mlp_hidden_dims_ood', dest='MLP_HIDDEN_DIMS_OOD', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions for the OOD MLP detectors.')
    parser.add_argument('--mlp_hidden_dims_coop', dest='MLP_HIDDEN_DIMS_COOP', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer dimensions for the COOP MLP classifiers.')

    parser.add_argument('--device', dest='DEVICE', type=str, default="cuda",
                    help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--k', dest='K_DETECTORS', type=int, default=3,
                        help='Number of detectors per DECOOP paper (often K=3).')
    parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int, default=256,
                    help='Batch size for training.')

    parser.add_argument('--lr_subfit', dest='LEARNING_SUBFIT', type=float, default=5e-2)
    parser.add_argument('--lr_prompt', dest='LEARNING_RATE_PROMPT', type=float, default=5e-4,
                        help='Learning rate for OOD detector training.')

    parser.add_argument('--n_epo', dest='NUM_EPOCHS', type=int, default=500,
                        help='Number of epochs for Detector training.')
    parser.add_argument('--n_eposub', dest='N_EPOCHS_SUBCLASSIFIER', type=int, default=100,
                        help='Number of epochs for Subclassifier training.')


    # OOD/ID Loss Regularization
    parser.add_argument("--OOD_MARGIN", type=float, default=0.2,
                        help="Margin for the entropy-based OOD loss.")
    parser.add_argument("--LAMBDA_ENTROPY", type=float, default=0.00,
                        help="Weight for the entropy regularization term in OOD prompt training.")
    parser.add_argument("--KL_COEFF", type=float, default=0.4,
                        help="Coefficient for KL divergence loss in COOP training.")

    # Conformal Prediction Thresholds
    parser.add_argument("--CP_OOD_THRESHOLD", type=float, default=0.9,
                    help="Conformal prediction OOD threshold. Will be calibrated during ECI.")
    parser.add_argument('--ALPHA_CP', type=float, default=0.1,
                        help='Alpha parameter for confidence penalty.')

    parser.add_argument('--WEIGHT_DECAY', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--WARMUP_EPOCHS', type=float, default=0.1,
                        help="Fraction of total epochs for linear warmup (e.g., 0.1 for 10% of epochs).")
    parser.add_argument("--LR_SCHEDULER_TYPE", type=str, default="cosine_with_warmup",
                    choices=["cosine_with_warmup", "plateau", "none"],
                    help="Type of LR scheduler to use.")
    # Plateau scheduler parameters
    parser.add_argument("--PLATEAU_FACTOR", type=float, default=0.1,
                        help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--PLATEAU_PATIENCE", type=int, default=5,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")


    args = parser.parse_args()

    return args