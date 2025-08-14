# train_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import time
import os
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)

# --- DATASET ---
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# --- DATA LOADING ---
def load_dataset(train_path, val_path):
    logger.info(f"Loading training data from {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    logger.info(f"Loading validation data from {val_path}")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    logger.info("Dataset loading completed")
    return train_text, val_text

# --- CHECKPOINT LOADING ---
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {path}")

# --- TRAINING LOGGING ---
def save_training_log(log_data, path):
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2)

def load_training_log(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

# --- TRAINING ---
def train_model(model, train_ids, val_ids, config, device,
                checkpoint_base, log_path, config_path,
                resume_from=None, max_batches_per_epoch=1000,
                patience=5):
    
    logger.info("Starting model training")
    training_log = load_training_log(log_path)
    train_ds = CharDataset(train_ids, config.block_size)
    val_ds = CharDataset(val_ids, config.block_size)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_epochs = 50
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if resume_from is not None:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        load_model(model, resume_from, device)
        try:
            start_epoch = int(resume_from.split("epoch")[-1].split(".")[0]) + 1
        except:
            pass

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, (xb, yb) in enumerate(train_loader):
            if max_batches_per_epoch is not None and i >= max_batches_per_epoch:
                break

            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / (i + 1)
        val_loss = evaluate(model, val_loader, device)
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch}/{num_epochs} completed in {elapsed:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")

        # Save log entry
        training_log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_perplexity": perplexity
        })
        save_training_log(training_log, log_path)

        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            ckpt_path = f"{checkpoint_base}{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"✅ Checkpoint saved (best so far): {ckpt_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"⚠️ No improvement in Val Loss for {epochs_without_improvement} epochs.")
            if epochs_without_improvement >= patience:
                logger.info(f"⏹ Early stopping triggered after {patience} epochs without improvement.")
                break

    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)


# --- EVALUATION ---
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- TEXT GENERATION ---
def generate_text(model, start_ids, max_new_tokens=100, device="cpu", temperature=1.0, top_k=None):
    model.eval()
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx[0].tolist()

# --- PERPLEXITY ---
def calculate_perplexity(model, data_ids, config, device, batch_size=32):
    model.eval()
    dataset = CharDataset(data_ids, config.block_size)
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)

    avg_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

# --- PLOT TRAINING ---
def plot_training_log(log_path="training_log.json", save_path="training_plot.png"):
    log = load_training_log(log_path)
    if not log:
        print("No log data found.")
        return

    epochs = [entry["epoch"] for entry in log]
    train_loss = [entry["train_loss"] for entry in log]
    val_loss = [entry["val_loss"] for entry in log]
    val_ppl = [entry["val_perplexity"] for entry in log]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.plot(epochs, val_ppl, label="Val Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Perplexity")
    plt.legend()
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
