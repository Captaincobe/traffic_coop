import torch
from torch.nn.functional import kl_div, softmax, log_softmax



# Helper for entropy
def calculate_batch_entropy_from_logits(logits):
    probs = softmax(logits, dim=1)
    log_probs = log_softmax(logits, dim=1)
    entropy_val = -torch.sum(probs * log_probs, dim=1)
    return torch.mean(entropy_val)

# Helper for KL Div
def kl_divergence_loss_from_logits(pred_logits, target_logits_detached):
    pred_log_probs = log_softmax(pred_logits, dim=1)
    target_probs_detached = softmax(target_logits_detached, dim=1).detach() # Ensure target is detached
    return kl_div(pred_log_probs, target_probs_detached, reduction='batchmean', log_target=False)
