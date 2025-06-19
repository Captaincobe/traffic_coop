import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear", "MLPClassifier", "MLMZeroShotClassifier"]


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, bias=False):
        super().__init__()
        self.dropout = dropout
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        out = torch.matmul(x, self.weight)
        return out + self.bias if self.bias is not None else out


class MLPClassifier(nn.Module):
    """
    Three-layer MLP identical to the original implementation.
    """
    def __init__(self, input_dim, hidden_dims, num_classes,
                 dropout=0.5, use_bn=True):
        if not hidden_dims:
            raise ValueError("`hidden_dims` must be non-empty.")
        nhid = hidden_dims[0]
        super().__init__()

        self.l1 = Linear(input_dim, nhid * 2, dropout, bias=True)
        self.l2 = Linear(nhid * 2, nhid, dropout, bias=True)
        self.l3 = Linear(nhid, num_classes, dropout, bias=True)

        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(nhid * 2)
            self.bn3 = nn.BatchNorm1d(nhid)

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.use_bn:
            x = self.bn3(x)
        x = F.relu(self.l3(x))   # keep ReLU to match original behaviour
        return x
    


class MLMZeroShotClassifier:
    def __init__(self, model_name, label_token_map, template="This traffic is [MASK].", device="cuda"):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.template = template
        self.label_token_map = label_token_map

        # ------------------------------------------------------------------
        # Convert label ➜ token‑id  ——  Scheme B: automatically add new
        # whole‑word tokens to the tokenizer when the default BPE split
        # would produce multiple sub‑tokens.  This keeps every verbalizer
        # one‑to‑one with a single token so that masked‑LM logits are
        # comparable across labels.
        # ------------------------------------------------------------------
        self.label_token_ids = {}

        # 1) Gather all words that need to be appended to the vocab
        tokens_to_add = []
        for word in label_token_map.values():
            if len(self.tokenizer.tokenize(word)) != 1:
                tokens_to_add.append(word)

        # 2) Add them *once* and resize the model’s embedding matrix
        if tokens_to_add:
            n_added = self.tokenizer.add_tokens(tokens_to_add, special_tokens=False)
            if n_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"[Info] Added {n_added} new tokenizer tokens for MLM labels: {tokens_to_add}")

        # 3) Build the final label→token‑id mapping (now guaranteed 1‑to‑1)
        for label, word in label_token_map.items():
            tokens = self.tokenizer.tokenize(word)
            if len(tokens) != 1:
                # Defensive fallback; should not happen after the addition step
                print(f"[Warning] Label '{label}' still maps to multiple tokens: {tokens}")
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
            self.label_token_ids[label] = token_id

    def eval(self):
        """
        Make the wrapped masked‑LM enter evaluation mode and
        return self so the call can be chained just like with
        nn.Module.eval().
        """
        self.model.eval()
        return self



    def predict(self, input_texts):
        """
        Parameters
        ----------
        input_texts : Union[str, List[str]]
            Raw flow‑text descriptions. A single string will be automatically converted
            to a one‑element list.

        Returns
        -------
        probs : torch.Tensor, shape = (B, C)
            Soft‑max probabilities over C labels for each sample in the batch.
        label_names : List[str]
            The label order corresponding to the second dimension of probs.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Assemble the prompt: "<flow text> This traffic is [MASK]."
        prompts = [
            f"{t} This traffic is {self.tokenizer.mask_token}."
            for t in input_texts
        ]

        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Locate the [MASK] position for every sample
        mask_positions = (batch.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        with torch.no_grad(): # MLMZeroShotClassifier 保持冻结，仅用于推理
            logits = self.model(**batch).logits        # (B, L, V)

        # Select logits corresponding to the [MASK] token
        mask_logits = logits[mask_positions]          # (B, V)

        # Keep only the logits of label verbalizer tokens
        label_token_ids = list(self.label_token_ids.values())
        scores = mask_logits[:, label_token_ids]      # (B, C)

        probs = torch.softmax(scores, dim=-1)         # (B, C)

        return probs.cpu(), list(self.label_token_ids.keys())

