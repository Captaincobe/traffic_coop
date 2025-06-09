import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Configuration and Hyperparameters for Model Training")

    dataset_choices = {
        "0": "ISCXVPN2016",
        "1": "ISCXTor2016",
        "2": "USTC-TFC2016",
        "3": "CIC-Darknet2020",
    }
    choices_help = ", ".join([f"{k} for {v}" for k, v in dataset_choices.items()])
    parser.add_argument('--dataset_name', type=str, default='ISCXVPN2016',
                        choices=["ISCXVPN2016", "ISCXTor2016", "USTC-TFC2016", "CIC-Darknet2020"],
                        help=f'Select dataset by number: {choices_help}')

    parser.add_argument('--llm', dest='LLM_MODEL_NAME', type=str, default="FacebookAI/roberta-base",
                        choices=["Qwen/Qwen2.5-7B-Instruct","bert-base-uncased","FacebookAI/roberta-base"],
                        help='Name of the pre-trained LLM model to use.')
    parser.add_argument('--device', dest='DEVICE', type=str, default="cuda",
                    help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int, default=256,
                    help='Batch size for training (LLMs often require smaller batch sizes).')
    
    parser.add_argument('--max_seq_length', dest='MAX_SEQ_LENGTH', type=int, default=128,
                    help='Maximum sequence length for LLM input.')
    
    parser.add_argument('--alpha_cp', dest='ALPHA_CP', type=float, default=0.1,
                        help='Alpha parameter for confidence penalty.')
    
    parser.add_argument('--cp_ood', dest='CP_OOD_THRESHOLD', type=float, default=0.1)
    
    parser.add_argument('--k', dest='K_DETECTORS', type=int, default=3,
                        help='Number of detectors per DECOOP paper (often K=3).')
    
    parser.add_argument('--lambda', dest='LAMBDA_ENTROPY', type=float, default=0.4,
                        help='Gamma parameter for OOD loss.')
    parser.add_argument('--margin', dest='OOD_MARGIN', type=float, default=0.4,
                    help='Gamma parameter for OOD loss.')
    parser.add_argument('--kl', dest='KL_COEFF', type=float, default=0.4)

    # parser.add_argument('--n_epochs_zs_classifier', dest='N_EPOCHS_ZS_CLASSIFIER', type=int, default=1,
    #                     help='Number of epochs for Zero-Shot Classifier training.')
    parser.add_argument('--n_epo', dest='NUM_EPOCHS', type=int, default=50,
                        help='Number of epochs for Detector training.')
    parser.add_argument('--n_epochs_subclassifier', dest='N_EPOCHS_SUBCLASSIFIER', type=int, default=50,
                        help='Number of epochs for Subclassifier training.')
    
    # parser.add_argument('--use_prompt_tuning', dest='USE_PROMPT_TUNING', action='store_true',
    #                     help='Enable prompt tuning to reduce trainable parameters.')
    # parser.add_argument('--prompt_tuning_length', dest='PROMPT_TUNING_LENGTH', type=int, default=5,
    #                     help='Number of prompt tokens for prompt tuning.')
    parser.add_argument('--weight_decay', dest='WEIGHT_DECAY', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    # parser.add_argument('--lr_llm', dest='LEARNING_RATE_LLM', type=float, default=1e-6,
    #                     help='Learning rate for the LLM backbone.')
    parser.add_argument('--lr_prompt', dest='LEARNING_RATE_PROMPT', type=float, default=5e-3,
                        help='Learning rate for the classification head.')
    # parser.add_argument('--lr_peft', dest='LEARNING_RATE_PEFT', type=float, default=1e-5,
    #                     help='Learning rate for PEFT parameters if used.')

    parser.add_argument('--num_base_classes', dest='NUM_BASE_CLASSES', type=int, default=-1,
                        help='Number of base classes. Set to -1 to be determined dynamically.')
    parser.add_argument('--num_all_classes', dest='NUM_ALL_CLASSES', type=int, default=-1,
                        help='Total number of all classes. Set to -1 to be determined dynamically.')

    # parser.add_argument('--ood_threshold_placeholder', dest='OOD_THRESHOLD_PLACEHOLDER', type=float, default=0.5,
    #                     help='Placeholder for OOD threshold; DECOOP typically uses Otsu method.')

    args = parser.parse_args()

    return args