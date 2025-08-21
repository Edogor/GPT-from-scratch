import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Union, Dict, Any
import math


class NeuralBigram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, use_output_layer=True, dropout=0.0):
        super(NeuralBigram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_output_layer = use_output_layer
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        if use_output_layer:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.fc = nn.Linear(embedding_dim, vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, logits=True):
        x = self.embedding(x)
        x = self.dropout(x)
        if self.use_output_layer:
            x = self.fc(x)
        if not logits:
            x = F.log_softmax(x, dim=-1)
        return x

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # (B, T_in) long
        *,
        max_new_tokens: int = 50,
        mode: str = "sample",  # sample or argmax
        eos_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,  # e.g. 50
        top_p: Optional[float] = None,  # e.g. 0.9
        repetition_penalty: float = 1.0,  # >1.0 discourages repeats
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate text from the model.

        Parameters:
        - input_ids: Tensor of shape (B, T_in) with input token IDs.
        - max_new_tokens: Maximum number of new tokens to generate.
        - mode: 'sample' for sampling, 'argmax' for greedy decoding.
        - eos_id: Optional end-of-sequence token ID.
        - temperature: Temperature for sampling.
        - top_k: Top-k filtering for sampling.
        - top_p: Top-p (nucleus) filtering for sampling.
        - repetition_penalty: Penalty factor for repeated tokens.
        - device: Device to run the generation on.

        Returns:
        - Generated token IDs of shape (B, T_out).
        """
        assert input_ids.dtype == torch.long, "input_ids must be LongTensor (token IDs)."

        was_training = self.training
        self.eval()

        device = device or next(self.parameters()).device
        x = input_ids.to(device)

        B = x.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # context: last token only (shape (B, 1))
            last_tok = x[:, -1:].to(device)

            # logits: (B, 1, V) -> take last step -> (B, V)
            logits = self(last_tok, logits=True)[:, -1, :]

            # repetition penalty (simple): down-weight tokens already present
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(B):
                    if not finished[b]:
                        seen = torch.unique(x[b])
                        logits[b, seen] -= math.log(repetition_penalty)

            # greedy argmax
            if mode == "argmax":
                next_ids = torch.argmax(logits, dim=-1)
            # else sample according after applying temperature, top-k, top-p
            else:
                # temperature scaling
                if temperature is not None and temperature != 1.0:
                    logits = logits / max(temperature, 1e-8)

                # top-k filter
                if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                    kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

                # top-p filter
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)

                    # mask tokens with cumulative prob > top_p
                    mask = cumprobs > top_p
                    # ensure at least one token remains
                    mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    # unsort back
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # If EOS provided, keep it once reached
            if eos_id is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)

            # append
            x = torch.cat([x, next_ids.unsqueeze(1)], dim=1)

            # update finished batch mask
            if eos_id is not None:
                finished = finished | (next_ids == eos_id)
                if torch.all(finished):
                    break

        if was_training:
            self.train()
        return x


# region Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, text_ids, block_size=128):
        super(TextDataset, self).__init__()
        self.text_ids = text_ids
        self.block_size = block_size

    def __len__(self):
        return (len(self.text_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.text_ids[start:end], dtype=torch.long)
        y = torch.tensor(self.text_ids[start + 1 : end + 1], dtype=torch.long)
        return x, y


def init_dataloader(data_ids, block_size=128, batch_size=64, train=True, shuffle=True):
    train_dataset = TextDataset(data_ids, block_size)
    if torch.cuda.is_available():
        num_workers = 4
        persistent_workers = True
        pin_memory = True
        prefetch_factor = 2
    else:
        num_workers = 0
        persistent_workers = False
        pin_memory = False
        prefetch_factor = None

    if train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader


# region utils
class WarmupThenCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, eta_min=0.0):
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = total_steps
        self.cosine = None
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        # initialize cosine on first post-warmup call
        if self.cosine is None:
            # remaining steps after warmup
            remain = max(1, self.total_steps - self.warmup_steps)
            self.cosine = CosineAnnealingLR(self.optimizer, T_max=remain, eta_min=self.eta_min)
            self.cosine.last_epoch = -1  # reset internal counter so first get_lr() is step 0
        return self.cosine.get_last_lr()

    def step(self, epoch=None):
        # Advance this scheduler
        super().step(epoch)
        # Also advance cosine if active and after warmup
        if self.cosine is not None:
            self.cosine.step(epoch if epoch is not None else None)
