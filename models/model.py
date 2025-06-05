import gc
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import softmax, cross_entropy
from transformers import AutoTokenizer,AutoModel, get_linear_schedule_with_warmup,AutoModelForSequenceClassification
from utils.eci_controller import ECIThresholdController

from utils.loss import calculate_batch_entropy_from_logits, kl_divergence_loss_from_logits
import torchvision
torchvision.disable_beta_transforms_warning()

# PEFT Configuration (Example LoRA)
USE_PEFT = True # Set to True to enable PEFT
peft_config = None

if USE_PEFT:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            # target_modules=["query", "value"] # BERT specific, check your LLM for appropriate modules
            # For other models, you might need to inspect model.named_modules() to find linear layers for LoRA
        )
    except ImportError:
        print("PEFT library not found. Running without PEFT.")
        USE_PEFT = False

# --- 2. Model Definitions ---
class LLMBasedTrafficClassifier(nn.Module):
    def __init__(self, llm_model_name, num_classes_output, freeze_llm_base=False,
                 use_prompt_tuning=False, prompt_length=0):
        super().__init__()
        import inspect
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 或者 "<|endoftext|>"，视模型而定

        # self.llm = AutoModelForSequenceClassification.from_pretrained(llm_model_name, pad_token_id=tokenizer.pad_token_id, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
        self.llm = AutoModelForSequenceClassification.from_pretrained(llm_model_name, num_labels=num_classes_output, pad_token_id=tokenizer.pad_token_id, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
        self.num_classes_output = num_classes_output # Number of classes this specific head outputs (e.g., NUM_BASE_CLASSES)
        self.accepts_token_type_ids = 'token_type_ids' in inspect.signature(self.llm.forward).parameters
        current_peft_config = peft_config
        self.use_prompt_tuning = use_prompt_tuning and prompt_length > 0
        self.prompt_length = prompt_length if self.use_prompt_tuning else 0
        if freeze_llm_base and not current_peft_config: # Freeze only if not using PEFT or if PEFT implies freezing
            for param in self.llm.parameters():
                param.requires_grad = False

        if current_peft_config and USE_PEFT:
            self.llm = get_peft_model(self.llm, current_peft_config)
            print(f"Applied PEFT to LLM. Trainable params:")
            self.llm.print_trainable_parameters()

        llm_hidden_size = self.llm.config.hidden_size
        if self.use_prompt_tuning:
            self.prompt_embeddings = nn.Parameter(torch.zeros(self.prompt_length, llm_hidden_size))
            nn.init.normal_(self.prompt_embeddings, std=0.02)
        self.classifier_head = nn.Linear(llm_hidden_size, num_classes_output)

    # def forward(self, input_ids, attention_mask, token_type_ids=None):
    #     if token_type_ids is not None and token_type_ids.nelement() > 0 and token_type_ids.numel() == input_ids.numel() : # Check if not empty and shape is compatible
    #          outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #     else:
    #          outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        
    #     pooled_output = outputs.last_hidden_state[:, 0] # CLS token embedding for BERT-like
    #     logits = self.classifier_head(pooled_output)
    #     return logits
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Remove any unexpected kwargs that might be passed from the training loop
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


        if self.use_prompt_tuning:
            batch_size = input_ids.size(0)
            prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
            extended_mask = torch.cat([torch.ones(batch_size, self.prompt_length, device=attention_mask.device), attention_mask], dim=1)
            if self.accepts_token_type_ids and token_type_ids is not None and token_type_ids.nelement() > 0:
                token_type_ids = torch.cat([torch.zeros(batch_size, self.prompt_length, dtype=token_type_ids.dtype, device=token_type_ids.device), token_type_ids], dim=1)
                outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=extended_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        else:
            if self.accepts_token_type_ids and token_type_ids is not None and token_type_ids.nelement() > 0 and token_type_ids.numel() == input_ids.numel():
                outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, 0].to(torch.float32) # CLS token embedding for BERT-like
        outputs = self.llm(**kwargs)
        pooled_output = outputs.logits
        logits = self.classifier_head(pooled_output)
        return logits

class LLMTrafficDECOOP:
    def __init__(self, args):
        self.llm_model_name = args.LLM_MODEL_NAME
        self.num_base_classes = args.NUM_BASE_CLASSES
        self.num_all_classes = args.NUM_ALL_CLASSES
        self.k_detectors = args.K_DECOOP_DETECTORS
        self.lr_llm = args.LEARNING_RATE_LLM
        self.lr_peft = args.LEARNING_RATE_PEFT
        self.lr_head = args.LEARNING_RATE_HEAD
        self.gamma_ood_loss = args.GAMMA_OOD_LOSS
        self.device = args.DEVICE
        self.batch_size = args.BATCH_SIZE
        self.ood_threshold_ = args.OOD_THRESHOLD_PLACEHOLDER
        self.eci_controller = ECIThresholdController(initial_threshold=args.OOD_THRESHOLD_PLACEHOLDER)
        self.n_epochs_detector = args.N_EPOCHS_DETECTOR
        self.n_epochs_zs = args.N_EPOCHS_ZS_CLASSIFIER
        self.n_epochs_subcls = args.N_EPOCHS_SUBCLASSIFIER
        self.peft_config = peft_config
        self.use_prompt_tuning = getattr(args, 'USE_PROMPT_TUNING', False)
        self.prompt_tuning_length = getattr(args, 'PROMPT_TUNING_LENGTH', 0)
        self.new_class_detectors_ = nn.ModuleList()
        self.sub_classifiers_ = nn.ModuleList()
        # ZS Classifier: freeze_llm_base=True if only tuning head/PEFT
        self.zs_classifier_ = LLMBasedTrafficClassifier(
            self.llm_model_name,
            self.num_base_classes,
            freeze_llm_base=True,
            use_prompt_tuning=self.use_prompt_tuning,
            prompt_length=self.prompt_tuning_length,
        ).to(self.device)
        self.base_class_global_indices_ = None

    def set_base_class_global_indices(self, base_class_global_indices):
        self.base_class_global_indices_ = sorted(list(set(base_class_global_indices)))
        if len(self.base_class_global_indices_) != self.num_base_classes:
            print(f"Warning: Length of provided base_class_global_indices ({len(self.base_class_global_indices_)}) "
                  f"does not match num_base_classes ({self.num_base_classes}). This can cause mapping issues.")


    def _create_optimizer_and_scheduler(self, model, num_training_steps):
        # Separate parameters for LLM backbone (if not fully frozen) and classifier head / PEFT params
        optimizer_grouped_parameters = []
        if hasattr(model, 'llm') and any(p.requires_grad for p in model.llm.parameters()):
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in model.llm.named_parameters() if p.requires_grad], 
                 'lr': self.lr_llm if not USE_PEFT else self.lr_peft}
            )
        if hasattr(model, 'classifier_head'):
            optimizer_grouped_parameters.append(
                {'params': model.classifier_head.parameters(), 'lr': self.lr_head}
            )
        
        if not optimizer_grouped_parameters: # If all frozen
            return None, None

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        return optimizer, scheduler
    
    def calibrate_eci_threshold(self, calibration_dataset, base_class_labels_set):
        """
        使用一组带标签的校准集（validation set）来更新 ECI 阈值
        :param calibration_dataset: Dataset 实例（含 label 字段）
        :param base_class_labels_set: set(int)，包含所有 base 类的全局索引
        """
        print("🧪 开始 ECI 阈值校准（Conformal）...")
        device = self.device
        calib_loader = DataLoader(calibration_dataset, batch_size=1, shuffle=False)
        updated_samples = 0

        for sample in calib_loader:
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            token_type_ids = sample.get('token_type_ids')
            if token_type_ids is not None and token_type_ids.nelement() > 0:
                token_type_ids = token_type_ids.to(device)
            else:
                token_type_ids = None

            # 获取 detector score（只使用一个 detector，也可以投票）
            with torch.no_grad():
                detector_scores = []
                for detector in self.new_class_detectors_:
                    logits = detector(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    score = torch.max(softmax(logits, dim=1)).item()
                    detector_scores.append(score)
                max_score = max(detector_scores)

            # 使用 ground truth 判断是否应被覆盖（是否是 base 类）
            gt_label = sample['labels'].item()  # 全局 label
            covered = gt_label in base_class_labels_set

            # 更新阈值
            self.eci_controller.update(max_score, covered)
            updated_samples += 1

            if updated_samples % 1500 == 0:
                print(f"已校准样本数：{updated_samples}，当前阈值：{self.eci_controller.get_threshold():.4f}，当前覆盖率：{self.eci_controller.get_coverage():.3f}")

        print(f"✅ ECI 校准完成，共更新样本数：{updated_samples}")
        print(f"最终 ECI 阈值：{self.eci_controller.get_threshold():.4f}，最终覆盖率估计：{self.eci_controller.get_coverage():.3f}")
    
    def _generic_train_loop(self, model, dataloader, loss_calculation_fn, num_epochs, model_description):
        model.train()
        num_training_steps = len(dataloader) * num_epochs
        optimizer, scheduler = self._create_optimizer_and_scheduler(model, num_training_steps)
        device = self.device
        if optimizer is None:
            print(f"Skipping training for {model_description} as no parameters require grad.")
            return

        for epoch in range(num_epochs):
            epoch_loss = 0
            processed_batches = 0
            for batch_idx, batch_data in enumerate(dataloader):
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                token_type_ids = batch_data.get('token_type_ids')
                if token_type_ids is not None and token_type_ids.nelement() > 0 : token_type_ids = token_type_ids.to(device)
                else: token_type_ids = None # Explicitly set to None if empty
                
                labels_local = batch_data['labels'].to(device) # These are local indices for base class models

                optimizer.zero_grad()
                # Only pass the required arguments to the model's forward method
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                # loss_calculation_fn is specific to ZS, Detector, or Sub-classifier
                # It might need more than just logits and labels (e.g., other model outputs or data parts for DECOOP losses)
                # For simplicity, we assume it can be called like this for now:
                # For Detector: loss_calculation_fn(logits_sim_base, labels_sim_base, logits_sim_new)
                # For Sub-classifier: loss_calculation_fn(logits_d_b_i, labels_d_b_i, logits_d_n_i, zs_detached_model, features_d_n_i)
                # This generic loop might need to be specialized for each component if loss_calculation_fn becomes too complex.
                
                # This is a simplification: loss_calculation_fn needs to handle DECOOP's complex structure
                # For a simple CE loss (like for ZS classifier):
                if model_description == "ZS Classifier": # ZS uses simple CE
                    loss = cross_entropy(logits, labels_local)
                else: # For Detectors and Sub-classifiers, loss_calculation_fn would be more complex
                      # This placeholder won't work for them without passing more args to loss_calculation_fn
                      # For now, let's assume it's passed within batch_data for complex losses (not ideal)
                    loss = loss_calculation_fn(logits, labels_local, model, batch_data) # model and batch_data passed for flexibility

                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
                epoch_loss += loss.item()
                processed_batches += 1
            
            if processed_batches > 0:
                print(f"Epoch {epoch+1}/{num_epochs}, {model_description} Avg Loss: {epoch_loss/processed_batches:.4f}")
        print(f"Finished training {model_description}")

    # --- Detailed Training for DECOOP Components ---
    def _train_detector_component(self, detector_model, train_dataset_base_local_labels, detector_idx, num_epochs):
        # 模拟“新类”（sim_new）与“基类”（sim_base）进行训练
        device = self.device
        batch_size = self.batch_size
        detector_model.train()
        num_training_steps = (len(train_dataset_base_local_labels) // batch_size) * num_epochs # Approximate
        optimizer, scheduler = self._create_optimizer_and_scheduler(detector_model, num_training_steps)

        # DECOOP: Create simulated base/new split for this detector
        # This partitioning should be more strategic (e.g., ensuring each base class is 'new' at least once across detectors)
        all_dataset_indices = np.arange(len(train_dataset_base_local_labels))
        np.random.shuffle(all_dataset_indices)
        # Example: Rotate which third is 'new'
        fold_size = len(all_dataset_indices) // self.k_detectors if self.k_detectors > 0 else len(all_dataset_indices) // 3
        if fold_size == 0 and len(all_dataset_indices)>0: fold_size = 1

        start_idx = detector_idx * fold_size
        end_idx = (detector_idx + 1) * fold_size if detector_idx < self.k_detectors -1 else len(all_dataset_indices)
        
        # This ensures that if k_detectors is e.g. 3, each third of the base classes acts as sim_new for one detector
        # This is a simplified way to achieve the paper's goal.
        sim_new_subset_indices = all_dataset_indices[start_idx:end_idx]
        sim_base_subset_indices = np.setdiff1d(all_dataset_indices, sim_new_subset_indices)

        if len(sim_base_subset_indices) == 0 or len(sim_new_subset_indices) == 0:
            print(f"Warning: Detector {detector_idx+1} - sim_base or sim_new subset is empty. Skipping training.")
            return

        sim_base_dataset = Subset(train_dataset_base_local_labels, sim_base_subset_indices)
        sim_new_dataset = Subset(train_dataset_base_local_labels, sim_new_subset_indices) # Labels are local base, but treated as "new" for entropy

        sim_base_loader = DataLoader(sim_base_dataset, batch_size=max(1,batch_size // 2), shuffle=True)
        sim_new_loader = DataLoader(sim_new_dataset, batch_size=max(1,batch_size // 2), shuffle=True) # Shuffle to get different pairings

        print(f"Training New Class Detector {detector_idx+1}/{self.k_detectors} (SimBase: {len(sim_base_dataset)}, SimNew: {len(sim_new_dataset)})...")
        for epoch in range(num_epochs):
            epoch_loss_val = 0; batches_done = 0
            # Iterate through sim_base_loader and try to get a matching batch from sim_new_loader
            # This ensures gradient updates are based on both parts of the loss
            iter_sim_new = iter(sim_new_loader)
            for batch_sim_base in sim_base_loader:
                try:
                    batch_sim_new = next(iter_sim_new)
                except StopIteration: # Reset new_loader if exhausted
                    iter_sim_new = iter(sim_new_loader)
                    try:
                        batch_sim_new = next(iter_sim_new)
                    except StopIteration: # If sim_new_loader is empty or exhausted even after reset
                        # print(f"SimNewLoader exhausted for epoch {epoch+1}. Skipping rest of SimBaseLoader.")
                        break 

                optimizer.zero_grad()
                
                # Simulated Base part
                logits_sim_base = detector_model(
                    input_ids=batch_sim_base['input_ids'].to(device),
                    attention_mask=batch_sim_base['attention_mask'].to(device),
                    token_type_ids=batch_sim_base.get('token_type_ids').to(device) if batch_sim_base.get('token_type_ids') is not None and batch_sim_base.get('token_type_ids').nelement() > 0 else None
                )
                labels_sim_base_local = batch_sim_base['labels'].to(device)
                loss_ce_sim_base = cross_entropy(logits_sim_base, labels_sim_base_local)
                entropy_sim_base = calculate_batch_entropy_from_logits(logits_sim_base)

                # Simulated New part
                logits_sim_new = detector_model(
                    input_ids=batch_sim_new['input_ids'].to(device),
                    attention_mask=batch_sim_new['attention_mask'].to(device),
                    token_type_ids=batch_sim_new.get('token_type_ids').to(device) if batch_sim_new.get('token_type_ids') is not None and batch_sim_new.get('token_type_ids').nelement() > 0 else None
                )
                entropy_sim_new = calculate_batch_entropy_from_logits(logits_sim_new)

                # DECOOP OOD Loss (Eq. 6)
                ood_penalty = torch.relu(self.gamma_ood_loss + entropy_sim_base - entropy_sim_new)
                total_loss = loss_ce_sim_base + ood_penalty
                
                total_loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
                epoch_loss_val += total_loss.item()
                batches_done +=1
            
            if batches_done > 0: print(f"  Epoch {epoch+1}, Detector {detector_idx+1} OOD Loss: {epoch_loss_val/batches_done:.4f}")
        # TODO: Implement Otsu threshold calculation for this detector after training
        # Store it or use it to update the global self.ood_threshold_
        print(f"Finished training Detector {detector_idx+1}")


    def _train_subclassifier_component(self, sub_classifier_model, corresponding_detector, 
                                     train_dataset_base_local_labels, zs_classifier_detached, 
                                     classifier_idx, num_epochs):
        sub_classifier_model.train()
        corresponding_detector.eval() # Detector is used for partitioning, not training here
        zs_classifier_detached.eval()
        batch_size = self.batch_size
        device = self.device

        num_training_steps = (len(train_dataset_base_local_labels) // batch_size) * num_epochs # Approximate
        optimizer, scheduler = self._create_optimizer_and_scheduler(sub_classifier_model, num_training_steps)
        if not optimizer: return

        # DECOOP: Partition data using the corresponding_detector and its (ideally Otsu-derived) threshold
        # This is a crucial step from the paper. For now, we'll use a placeholder threshold.
        # We need to iterate through train_dataset_base_local_labels, get detector scores, and split.
        
        d_b_i_indices = [] # Indices for samples classified as "base" by this detector
        d_n_i_indices = [] # Indices for samples classified as "new" by this detector (for KL loss)

        temp_eval_loader = DataLoader(train_dataset_base_local_labels, batch_size=batch_size, shuffle=False)
        all_original_indices = [] # To map subset indices back to original dataset indices

        with torch.no_grad():
            offset = 0
            for batch_data in temp_eval_loader:
                logits_detector = corresponding_detector(
                    input_ids=batch_data['input_ids'].to(device),
                    attention_mask=batch_data['attention_mask'].to(device),
                    token_type_ids=batch_data.get('token_type_ids').to(device) if batch_data.get('token_type_ids') is not None and batch_data.get('token_type_ids').nelement() > 0 else None
                )
                # Using max softmax prob as "base score" for detector (higher = more likely base)
                base_scores = torch.max(softmax(logits_detector, dim=1), dim=1)[0].cpu().numpy()
                
                batch_size_current = batch_data['input_ids'].size(0)
                original_indices_batch = np.arange(offset, offset + batch_size_current)
                offset += batch_size_current

                for i, score in enumerate(base_scores):
                    current_original_idx = original_indices_batch[i]
                    if score >= self.ood_threshold_: # Placeholder threshold
                        d_b_i_indices.append(current_original_idx)
                    else:
                        d_n_i_indices.append(current_original_idx)
        
        if not d_b_i_indices and not d_n_i_indices: # If both empty, dataset was likely empty
             print(f"Warning: Sub-classifier {classifier_idx+1} - both D_b_i and D_n_i are empty. Skipping.")
             return
        if not d_b_i_indices:
            print(f"Warning: Sub-classifier {classifier_idx+1} - D_b_i is empty. CE loss part will be skipped.")
        if not d_n_i_indices:
            print(f"Warning: Sub-classifier {classifier_idx+1} - D_n_i is empty. KL loss part will be skipped.")


        d_b_i_dataset = Subset(train_dataset_base_local_labels, d_b_i_indices) if d_b_i_indices else None
        d_n_i_dataset = Subset(train_dataset_base_local_labels, d_n_i_indices) if d_n_i_indices else None

        d_b_i_loader = DataLoader(d_b_i_dataset, batch_size=max(1,batch_size // 2), shuffle=True) if d_b_i_dataset else None
        d_n_i_loader = DataLoader(d_n_i_dataset, batch_size=max(1,batch_size // 2), shuffle=True) if d_n_i_dataset else None

        print(f"Training Sub-classifier {classifier_idx+1}/{self.k_detectors} (D_b_i: {len(d_b_i_dataset) if d_b_i_dataset else 0}, D_n_i: {len(d_n_i_dataset) if d_n_i_dataset else 0})...")

        for epoch in range(num_epochs):
            epoch_loss_val = 0; batches_done = 0
            
            # Handle CE loss part (D_b_i)
            if d_b_i_loader:
                for batch_d_b_i in d_b_i_loader:
                    optimizer.zero_grad()
                    logits_sub = sub_classifier_model(
                        input_ids=batch_d_b_i['input_ids'].to(device),
                        attention_mask=batch_d_b_i['attention_mask'].to(device),
                        token_type_ids=batch_d_b_i.get('token_type_ids').to(device) if batch_d_b_i.get('token_type_ids') is not None and batch_d_b_i.get('token_type_ids').nelement() > 0 else None
                    )
                    labels_local = batch_d_b_i['labels'].to(device)
                    loss_ce = cross_entropy(logits_sub, labels_local)
                    loss_ce.backward() # Accumulate gradients if KL part exists
                    epoch_loss_val += loss_ce.item()
                    batches_done +=1 # Count this as a "loss contributing" batch part

            # Handle KL loss part (D_n_i)
            if d_n_i_loader:
                for batch_d_n_i in d_n_i_loader:
                    # If CE loss was already computed, this grad zero might be redundant or harmful if not careful
                    # optimizer.zero_grad() # Careful here if accumulating grads
                    
                    input_ids_dn = batch_d_n_i['input_ids'].to(device)
                    attn_mask_dn = batch_d_n_i['attention_mask'].to(device)
                    token_type_ids_dn = batch_d_n_i.get('token_type_ids')
                    if token_type_ids_dn is not None and token_type_ids_dn.nelement() > 0 : token_type_ids_dn = token_type_ids_dn.to(device)
                    else: token_type_ids_dn = None

                    logits_sub_for_kl = sub_classifier_model(input_ids=input_ids_dn, attention_mask=attn_mask_dn, token_type_ids=token_type_ids_dn)
                    with torch.no_grad():
                        target_zs_logits = zs_classifier_detached(input_ids=input_ids_dn, attention_mask=attn_mask_dn, token_type_ids=token_type_ids_dn)
                    
                    loss_kl = kl_divergence_loss_from_logits(logits_sub_for_kl, target_zs_logits)
                    loss_kl.backward() # Accumulate gradients
                    epoch_loss_val += loss_kl.item() 
                    batches_done +=1 # Count this as a "loss contributing" batch part
            
            if batches_done > 0: # Only step if some loss was computed and backwarded
                 optimizer.step()
                 if scheduler: scheduler.step()
                 print(f"  Epoch {epoch+1}, Sub-classifier {classifier_idx+1} Combined Loss: {epoch_loss_val/batches_done:.4f}")
        print(f"Finished training Sub-classifier {classifier_idx+1}")

    def fit(self, train_dataset_base_local_labels):

        device = self.device
        if self.base_class_global_indices_ is None:
            raise ValueError("Base class global indices not set. Call `set_base_class_global_indices` first.")

        # 1. Train ZS Classifier （全体样本 + 冻结LLM）
        print("Training ZS/General LLM Classifier...")
        zs_dataloader = DataLoader(train_dataset_base_local_labels, batch_size=self.batch_size, shuffle=True)
        
        def zs_loss_fn(logits, labels, model=None, batch_data=None): # model and batch_data not used for simple CE
            return cross_entropy(logits, labels)
        self._generic_train_loop(self.zs_classifier_, zs_dataloader, zs_loss_fn, self.n_epochs_zs, "ZS Classifier")
        
        # Detached ZS model for sub-classifier's KL divergence target
        # zs_classifier_detached = LLMBasedTrafficClassifier(self.llm_model_name, self.num_base_classes, freeze_llm_base=True).to(device)
        zs_classifier_detached = LLMBasedTrafficClassifier(
            self.llm_model_name,
            self.num_base_classes,
            freeze_llm_base=True,
            use_prompt_tuning=self.use_prompt_tuning,
            prompt_length=self.prompt_tuning_length,
        ).to(device)
        zs_classifier_detached.load_state_dict(self.zs_classifier_.state_dict()) # Copy weights
        zs_classifier_detached.eval()

        # 2. Train K New Class Detectors
        for i in range(self.k_detectors):
            # detector = LLMBasedTrafficClassifier(self.llm_model_name, self.num_base_classes, freeze_llm_base=False).to(device) # Detectors might need to fine-tune LLM more
            detector = LLMBasedTrafficClassifier(
                self.llm_model_name,
                self.num_base_classes,
                freeze_llm_base=False,
                use_prompt_tuning=self.use_prompt_tuning,
                prompt_length=self.prompt_tuning_length,
            ).to(device) # Detectors might need to fine-tune LLM more
            self._train_detector_component(detector, train_dataset_base_local_labels, i, self.n_epochs_detector)
            self.new_class_detectors_.append(detector)
        
        # TODO: Implement actual Otsu threshold calculation based on all detectors' outputs
        # For now, self.ood_threshold_ is a placeholder.
        print(f"Using OOD Threshold (placeholder/fixed): {self.ood_threshold_}")

        # 3. Train K Sub-Classifiers
        for i in range(self.k_detectors):
            # sub_classifier = LLMBasedTrafficClassifier(self.llm_model_name, self.num_base_classes, freeze_llm_base=True).to(device)
            sub_classifier = LLMBasedTrafficClassifier(
                self.llm_model_name,
                self.num_base_classes,
                freeze_llm_base=True,
                use_prompt_tuning=self.use_prompt_tuning,
                prompt_length=self.prompt_tuning_length,
            ).to(device)
            # Potentially initialize sub-classifier's LLM from corresponding detector's LLM
            if i < len(self.new_class_detectors_) and hasattr(self.new_class_detectors_[i], 'llm'):
                sub_classifier.llm.load_state_dict(self.new_class_detectors_[i].llm.state_dict())
            
            if i < len(self.new_class_detectors_): # Ensure detector exists
                self._train_subclassifier_component(sub_classifier, self.new_class_detectors_[i], 
                                                  train_dataset_base_local_labels, zs_classifier_detached, 
                                                  i, self.n_epochs_subcls)
                self.sub_classifiers_.append(sub_classifier)
            else: # Fallback for safety if a detector failed to instantiate
                print(f"Detector {i} not found, adding dummy sub-classifier.")
                self.sub_classifiers_.append(sub_classifier) # Add minimally initialized

        print("LLMTrafficDECOOP model training complete.")
        gc.collect() # Clean up memory
        torch.cuda.empty_cache() # if using CUDA


    def predict_proba(self, tokenized_sample_dict): # input is a dict from LLMCSVTrafficDataset
        self.zs_classifier_.eval()
        for model in self.new_class_detectors_: model.eval()
        for model in self.sub_classifiers_: model.eval()
        device=self.device
        input_ids = tokenized_sample_dict['input_ids'].unsqueeze(0).to(device) # Add batch dim
        attention_mask = tokenized_sample_dict['attention_mask'].unsqueeze(0).to(device)
        token_type_ids = tokenized_sample_dict.get('token_type_ids')
        if token_type_ids is not None and token_type_ids.nelement() > 0:
            token_type_ids = token_type_ids.unsqueeze(0).to(device)
        else: token_type_ids = None
            
        detector_base_scores = []
        with torch.no_grad():
            for detector in self.new_class_detectors_:
                logits_detector = detector(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # Score: max softmax prob among base classes known to the detector
                score = torch.max(softmax(logits_detector, dim=1)).item() 
                detector_base_scores.append(score)
        
        final_probs_base_local_np = None # Probabilities for local base class indices (0 to NUM_BASE_CLASSES-1)

        if not detector_base_scores or not self.sub_classifiers_: # Fallback if no detectors/subclassifiers trained
            # print("DEBUG: No detectors/subclassifiers or issue, falling back to ZS.")
            with torch.no_grad():
                zs_logits = self.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                final_probs_base_local_np = softmax(zs_logits, dim=1).squeeze().cpu().numpy()
        # elif np.max(detector_base_scores) < self.ood_threshold_: # Classified as NEW (Eq. 8)
        elif not self.eci_controller.is_in_conformal_set(detector_base_scores):
            # print("DEBUG: Classified as NEW by OOD detectors -> ZS Classifier")
            with torch.no_grad():
                zs_logits = self.zs_classifier_(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                final_probs_base_local_np = softmax(zs_logits, dim=1).squeeze().cpu().numpy()
        else: # Classified as BASE
            best_detector_idx = np.argmax(detector_base_scores)
            # print(f"DEBUG: Classified as BASE by detector {best_detector_idx} -> Sub-classifier {best_detector_idx}")
            selected_sub_classifier = self.sub_classifiers_[best_detector_idx]
            with torch.no_grad():
                sub_logits = selected_sub_classifier(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                final_probs_base_local_np = softmax(sub_logits, dim=1).squeeze().cpu().numpy()
        
        # Ensure numpy array and 1D
        if final_probs_base_local_np is not None and not isinstance(final_probs_base_local_np, np.ndarray):
            final_probs_base_local_np = np.array([final_probs_base_local_np]) 
        elif final_probs_base_local_np is not None and final_probs_base_local_np.ndim == 0 and self.num_base_classes == 1: # scalar to 1d array
            final_probs_base_local_np = np.array([final_probs_base_local_np.item()])


        output_probas_all_classes = np.zeros(self.num_all_classes)
        if final_probs_base_local_np is not None and self.base_class_global_indices_ is not None:
            if len(final_probs_base_local_np) == len(self.base_class_global_indices_):
                for local_idx, prob_val in enumerate(final_probs_base_local_np):
                    global_idx = self.base_class_global_indices_[local_idx]
                    if 0 <= global_idx < self.num_all_classes:
                         output_probas_all_classes[global_idx] = prob_val
            else:
                # This case usually means final_probs_base_local_np is not correctly shaped or num_base_classes mismatch
                print(f"Warning: Prob vector length ({len(final_probs_base_local_np) if final_probs_base_local_np is not None else 'None'}) "
                      f"mismatch with num base class indices ({len(self.base_class_global_indices_)}). Or num_base_classes = {self.num_base_classes}. Using fallback.")
                # Fallback if shapes are mismatched (e.g. if num_base_classes=0 or 1 and array is scalar)
                # Default to uniform if major issue
                output_probas_all_classes = np.ones(self.num_all_classes) / self.num_all_classes


        current_sum = np.sum(output_probas_all_classes)
        if current_sum == 0: return np.ones(self.num_all_classes) / self.num_all_classes
        return output_probas_all_classes / current_sum
