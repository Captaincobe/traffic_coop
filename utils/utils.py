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


def debug_model_training(train_dataset, model_instance, base_class_indices_num_sorted):
    print("\n========== [DEBUG MODE STARTED] ==========")

    # 1. 检查标签映射是否正确
    print("\n[1] 标签映射检查:")
    labels = [sample['labels'].item() for sample in train_dataset]
    global_labels = [sample['global_labels'].item() for sample in train_dataset]
    print(f"- 总样本数: {len(labels)}")
    print(f"- 本地标签集合: {sorted(set(labels))}")
    print(f"- 全局标签集合: {sorted(set(global_labels))}")
    print(f"- Base class 索引集合: {sorted(base_class_indices_num_sorted)}")

    # 2. 检查 base class 是否被包含在训练集中
    print("\n[2] base class 覆盖情况:")
    for cls_id in base_class_indices_num_sorted:
        count = global_labels.count(cls_id)
        print(f" - 类别 {cls_id}: 样本数 = {count}")
        if count == 0:
            print("   ⚠️ 警告: 此类别未出现在训练集中！")

    # 3. 检查模型中参与训练的参数
    print("\n[3] 模型参数训练状态检查:")
    for name, param in model_instance.named_parameters():
        if param.requires_grad:
            print(f"✅ 可训练: {name} - {tuple(param.shape)}")
        else:
            print(f"❌ 冻结: {name} - {tuple(param.shape)}")

    # 4. 检查 prompt 输出的 logits 是否变化，entropy 是否正常
    from torch.nn.functional import softmax
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(model_instance.device)
    attention_mask = batch['attention_mask'].to(model_instance.device)
    labels = batch['labels'].to(model_instance.device)

    k = 0  # detector index

    model_instance.eval()
    with torch.no_grad():
        logits = model_instance.prompt_manager.forward_ood(k=k, input_ids=input_ids, attention_mask=attention_mask)
        probs = softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        print("\n[4] Prompt 输出分布 & Entropy 检查:")
        print(f" - Logits 均值: {logits.mean().item():.4f}, 标准差: {logits.std().item():.4f}")
        print(f" - Entropy 范围: min={entropy.min().item():.4f}, max={entropy.max().item():.4f}")
        print(f" - 样本预测分布:\n{probs.cpu().numpy()}")

    print("\n========== [DEBUG MODE ENDED] ==========")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta: # 如果分数没有显著提升
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # 如果分数有显著提升
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            # 保存整个模型的state_dict，或者只保存需要训练的prompt和classifier
            # 考虑到LLMTrafficDECOOP的结构，保存其state_dict可能最简单
            self.best_model_state = model.state_dict()
            self.val_loss_min = val_loss
            