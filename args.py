import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Configuration and Hyperparameters for Model Training")
    parser.add_argument('--num_base_classes', dest='NUM_BASE_CLASSES', type=int, default=-1)
    parser.add_argument('--num_all_classes', dest='NUM_ALL_CLASSES', type=int, default=-1)
    parser.add_argument('--label_map', dest='LABEL_TOKEN_MAP', default=
    {
        "AUDIO": "audio",         
        "VIDEO": "video",         
        "FILE-TRANSFER": "file",         
        "BROWSING": "web",         
        "CHAT": "chat",           
        "VOIP": "call",            
        "P2P": "sharing",          
        "MAIL": "mail",          
        "STREAMING": "stream",      
        "VPN-BROWSING": "vpnweb",  
        "VPN-CHAT": "vpnchat",     
        "VPN-FT": "vpnftp",        
        "VPN-MAIL": "vpnmail",    
        "VPN-P2P": "vpnshare",     
        "VPN-STREAMING": "vpnvideo", 
        "VPN-VOIP": "vpncall",     
        "FT": "ftp"
    })
    parser.add_argument('--dataset_name', type=str, default='ISCXVPN2016',
                        choices=["ISCXVPN2016", "ISCXTor2016", "USTC-TFC2016", "CIC-Darknet2020"])
    # Few-shot training parameter
    parser.add_argument('--samples_per_class', dest='SAMPLES_PER_CLASS', type=int, default=200,
                        help='Number of samples per base class for few-shot training.')

    parser.add_argument('--LLM_MODEL_NAME', type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased","FacebookAI/roberta-base"],
                        help='Name of the pre-trained LLM model to use.')
    parser.add_argument('--device', dest='DEVICE', type=str, default="cuda",
                    help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--max_seq_length', dest='MAX_SEQ_LENGTH', type=int, default=128,
                    help='Maximum sequence length for LLM input.') # 1: 128, 
    parser.add_argument('--prompt_len', dest='PROMPT_LENGTH', type=int, default=30)
    parser.add_argument('--k', dest='K_DETECTORS', type=int, default=3,
                        help='Number of detectors per DECOOP paper (often K=3).')
    parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int, default=256,
                    help='Batch size for training (LLMs often require smaller batch sizes).')
    
    parser.add_argument('--lr_subfit', dest='LEARNING_SUBFIT', type=float, default=5e-5)
    parser.add_argument('--lr_prompt', dest='LEARNING_RATE_PROMPT', type=float, default=5e-3,
                        help='Learning rate for the classification head.')

    parser.add_argument('--n_epo', dest='NUM_EPOCHS', type=int, default=500,
                        help='Number of epochs for Detector training.')
    parser.add_argument('--n_eposub', dest='N_EPOCHS_SUBCLASSIFIER', type=int, default=50,
                        help='Number of epochs for Subclassifier training.')


    # OOD/ID Loss Regularization
    parser.add_argument("--OOD_MARGIN", type=float, default=0.2,
                        help="Margin for the entropy-based OOD loss.")
    parser.add_argument("--LAMBDA_ENTROPY", type=float, default=0.05, # 熵正则化系数，尝试0.01到0.1
                        help="Weight for the entropy regularization term in OOD prompt training.")
    parser.add_argument("--KL_COEFF", type=float, default=0.4, # KL散度损失系数，用于COOP训练
                        help="Coefficient for KL divergence loss in COOP prompt training.")

    #  Conformal Prediction Thresholds
    parser.add_argument("--CP_OOD_THRESHOLD", type=float, default=0.9, # 这是一个校准后的值，这里是初始默认值
                    help="Conformal prediction OOD threshold. Will be calibrated during ECI.")
    parser.add_argument('--ALPHA_CP', type=float, default=0.1,
                        help='Alpha parameter for confidence penalty.')


    parser.add_argument('--WARMUP_EPOCHS', type=float, default=0.1,
                        help="Fraction of total epochs for linear warmup (e.g., 0.1 for 10% of epochs).")
    parser.add_argument("--LR_SCHEDULER_TYPE", type=str, default="cosine_with_warmup",
                    choices=["cosine_with_warmup", "plateau", "none"],
                    help="Type of LR scheduler to use. 'cosine_with_warmup' is generally recommended for Transformers.")
    # Plateau 调度器参数
    parser.add_argument("--PLATEAU_FACTOR", type=float, default=0.1,
                        help="Factor by which the learning rate will be reduced. new_lr = lr * factor.")
    parser.add_argument("--PLATEAU_PATIENCE", type=int, default=5,
                        help="Number of epochs with no improvement after which learning rate will be reduced (if LR_SCHEDULER_TYPE is 'plateau').")


    args = parser.parse_args()

    return args

    # parser.add_argument('--weight_decay', dest='WEIGHT_DECAY', type=float, default=5e-4,
    #             help='Weight decay (L2 loss on parameters).')