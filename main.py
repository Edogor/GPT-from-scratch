# main.py
import torch
import logging
import os
import json
from datetime import datetime

from bpe_tokenizer import BPETokenizer
from gpt_model import GPTConfig, GPTModel
from train_utils import (
    load_dataset,
    train_model,
    calculate_perplexity,
    plot_training_log,
    load_model
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- SETTINGS ---
train_path = "../corpora/Shakespeare_clean_train.txt"
val_path = "../corpora/Shakespeare_clean_valid.txt"
num_merges = 50
mode = "train"  # "train" oder "generate"
resume = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- STEP 1: Load text ---
logger.info("Step 1: Loading text data and training BPE")
train_text, val_text = load_dataset(train_path, val_path)
logger.info("Dataset loading completed")

# --- MODE HANDLING ---
if mode == "train":
    logger.info("Step 5: Training mode selected")

    if not os.path.exists("runs"):
        os.makedirs("runs")

    
    # Automatisch generieren:
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_k{num_merges}"
    #Manuell setzen (optional): Kommentar entfernen und oben auskommentieren
    #run_name = "2025-08-03_17-25_k200"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)

    logger.info(f"Training BPE tokenizer")
    tokenizer = BPETokenizer(num_merges=num_merges)
    tokenizer.train(train_text)
    tokenizer.save(os.path.join(run_dir, "tokenizer", "bpe_merges.txt"))
    tokenizer.save_vocab(os.path.join(run_dir, "tokenizer", "bpe_vocab.txt"))
    logger.info("BPE tokenizer training completed")

    logger.info("Step 2: Tokenizing text data")
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    logger.info("Step 2 completed: Tokenization finished")

    # GPT Config
    vocab_size = len(tokenizer.vocab)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )

    model = GPTModel(config)
    model.to(device)
    logger.info("Step 3: Updating GPT configuration")
    logger.info("GPT model initialization completed")
    logger.info("Step 3 completed: GPT configuration updated")

    checkpoint_base = os.path.join(run_dir, "gpt_epoch")
    log_path = os.path.join(run_dir, "training_log.json")
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f)

    # Resume?
    resume_from = None
    if resume:
        ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("gpt_epoch") and f.endswith(".pt")])
        if ckpts:
            resume_from = os.path.join(run_dir, ckpts[-1])
        else:
            logger.warning(f"⚠️ Checkpoint {resume_from} nicht gefunden – starte Training neu.")

    train_model(
        model, train_ids, val_ids, config, device,
        checkpoint_base=checkpoint_base,
        log_path=log_path,
        config_path=config_path,
        resume_from=resume_from,
        max_batches_per_epoch=1000,
        patience=3
    )


elif mode == "generate":
    logger.info("Step 5: Generation mode selected")

    if not os.path.exists("runs"):
        raise FileNotFoundError("⚠️ Kein 'runs' Ordner vorhanden – bitte zuerst trainieren.")

    # Finde passenden Run
    matching_runs = sorted([r for r in os.listdir("runs") if f"k{num_merges}" in r])
    if not matching_runs:
        raise ValueError(f"⚠️ Kein Run mit k={num_merges} gefunden!")
    run_name = matching_runs[-1]
    run_dir = os.path.join("runs", run_name)
    logger.info(f"📂 Verwende Run: {run_name}")

    # Lade Tokenizer
    tokenizer = BPETokenizer(num_merges=num_merges)
    tokenizer.load(os.path.join(run_dir, "tokenizer", "bpe_merges.txt"))
    tokenizer.load_vocab(os.path.join(run_dir, "tokenizer", "bpe_vocab.txt"))

    # Lade Config
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    config = GPTConfig(**cfg)

    # Lade Modell + Checkpoint
    model = GPTModel(config)
    model.to(device)
    ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("gpt_epoch") and f.endswith(".pt")])
    if not ckpts:
        raise FileNotFoundError("⚠️ Kein Modell-Checkpoint gefunden!")
    last_ckpt = os.path.join(run_dir, ckpts[-1])
    load_model(model, last_ckpt, device)

    # --- Text generieren ---
    start_ids = [tokenizer.token2id.get("<s>", 0)]
    sample = model.generate(start_ids, max_new_tokens=100, temperature=0.9, top_k=20)
    text = tokenizer.decode(sample)

    print("\n" + "=" * 40 + "\nGenerated Text:\n" + "=" * 40 + f"\n{text}\n")

    # --- Perplexity berechnen ---
    val_ids = tokenizer.encode(val_text)
    ppl = calculate_perplexity(model, val_ids, config, device)
    print(f"Validation Perplexity: {ppl:.2f}")
    plot_training_log(os.path.join(run_dir, "training_log.json"))

else:
    logger.warning("Unknown mode. Set mode to either 'train' or 'generate'.")
