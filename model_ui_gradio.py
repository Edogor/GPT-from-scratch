import gradio as gr
import os
from datetime import datetime
from gpt_model import GPTConfig, GPTModel
from train_utils import load_dataset, train_model, load_model, plot_training_log
from bpe_tokenizer import BPETokenizer
import torch
import json
import pandas as pd

TRAIN_PATH = "../corpora/Shakespeare_clean_train.txt"
VAL_PATH = "../corpora/Shakespeare_clean_valid.txt"

# --- TRAINING LOGIK ---
def start_training(k_merges, n_layer, n_embd, n_head, block_size, dropout):
    n_head = int(n_head)  # von Dropdown kommt als String
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{timestamp}_k{k_merges}_L{n_layer}_E{n_embd}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)

    train_text, val_text = load_dataset(TRAIN_PATH, VAL_PATH)
    tokenizer = BPETokenizer(num_merges=k_merges)
    tokenizer.train(train_text)
    tokenizer.save(os.path.join(run_dir, "tokenizer", "bpe_merges.txt"))
    tokenizer.save_vocab(os.path.join(run_dir, "tokenizer", "bpe_vocab.txt"))
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    vocab_size = len(tokenizer.vocab)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPTModel(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint_base = os.path.join(run_dir, "gpt_epoch")
    log_path = os.path.join(run_dir, "training_log.json")
    config_path = os.path.join(run_dir, "config.json")

    resume_from = None
    checkpoints = sorted([f for f in os.listdir(run_dir) if f.startswith("gpt_epoch") and f.endswith(".pt")])
    if checkpoints:
        last_ckpt = checkpoints[-1]
        resume_from = os.path.join(run_dir, last_ckpt)

    train_model(
        model, train_ids, val_ids, config, device,
        checkpoint_base=checkpoint_base,
        log_path=log_path,
        config_path=config_path,
        resume_from=resume_from,
        max_batches_per_epoch=1000
    )

    return f"✅ Training abgeschlossen: {run_name}"

# --- RUN-ÜBERSICHT ---
def get_run_info():
    runs = []
    for run in sorted(os.listdir("runs")):
        run_dir = os.path.join("runs", run)
        try:
            with open(os.path.join(run_dir, "config.json")) as f:
                cfg = json.load(f)
            with open(os.path.join(run_dir, "training_log.json")) as f:
                log = json.load(f)
            if not log: continue
            last = log[-1]
            runs.append({
                "run": run,
                "epoch": last["epoch"],
                "train_loss": round(last["train_loss"], 3),
                "val_loss": round(last["val_loss"], 3),
                "ppl": round(last["val_perplexity"], 2),
                "n_layer": cfg["n_layer"],
                "n_embd": cfg["n_embd"],
                "block_size": cfg["block_size"],
            })
        except: continue
    return pd.DataFrame(runs)

selected_run = gr.State("")

with gr.Blocks() as demo:
    with gr.Tab("📦 Neues Modell erstellen & trainieren"):
        gr.Markdown("### GPT-Modell starten")

        k_merges = gr.Slider(100, 1000, value=200, step=50, label="BPE Merges (k)")
        n_layer = gr.Slider(1, 12, value=4, step=1, label="Anzahl Layer")
        n_embd = gr.Slider(64, 512, value=128, step=32, label="Embedding-Dimension")
        n_head = gr.Dropdown(choices=["1", "2", "4", "8"], value="4", label="Anzahl Attention-Heads")
        block_size = gr.Slider(32, 256, value=64, step=32, label="Blockgröße (Kontext)")
        dropout = gr.Slider(0.0, 0.3, value=0.1, step=0.05, label="Dropout")

        start_btn = gr.Button("🚀 Training starten")
        output = gr.Textbox(label="Status")

        def update_heads(n_embd_value):
            valid_heads = [str(h) for h in range(1, n_embd_value + 1) if n_embd_value % h == 0]
            default = "4" if "4" in valid_heads else valid_heads[0]
            return gr.Dropdown.update(choices=valid_heads, value=default)

        n_embd.change(update_heads, inputs=n_embd, outputs=n_head)

        start_btn.click(
            start_training,
            inputs=[k_merges, n_layer, n_embd, n_head, block_size, dropout],
            outputs=output
        )


    with gr.Tab("📂 Vorhandene Runs"):
        gr.Markdown("### Wähle ein vorhandenes Modell")
        df = gr.Dataframe(headers=["run", "epoch", "train_loss", "val_loss", "ppl", "n_layer", "n_embd", "block_size"], interactive=False, label="Modellübersicht")
        refresh_btn = gr.Button("🔄 Liste aktualisieren")
        selected_text = gr.Textbox(label="Gewählter Run (zum Kopieren)", interactive=False)

        def refresh_table():
            table = get_run_info()
            return table.values.tolist()

        def select_row(evt: gr.SelectData):
            row_idx = evt.index[0]
            table = get_run_info()
            try:
                run_name = table.iloc[row_idx]["run"]
                return run_name
            except:
                return ""

        refresh_btn.click(refresh_table, outputs=df)
        df.select(select_row, outputs=selected_text)
        selected_text.change(lambda s: s, inputs=selected_text, outputs=selected_run)

        with gr.Tab("🧠 Text generieren"):
            gr.Markdown("### Wähle ein Modell und generiere Text")

            with gr.Row():
                selected_gen_run = gr.Dropdown(
                    choices=sorted(os.listdir("runs")), 
                    label="Modell wählen (Run-Ordner)"
                )
                refresh_gen_btn = gr.Button("🔄")

            prompt_input = gr.Textbox(label="Prompt (optional)", placeholder="z. B. Once upon a time")
            max_tokens = gr.Slider(10, 500, value=100, step=10, label="Max neue Tokens")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=20, step=1, label="Top-k (0 = deaktiviert)")

            gen_btn = gr.Button("🎬 Text generieren")
            gen_output = gr.Textbox(label="Output")

            def list_runs():
                return gr.Dropdown.update(choices=sorted(os.listdir("runs")))

            def generate_text_from_model(run_name, prompt, max_tokens, temperature, top_k):
                if not run_name:
                    return "❌ Kein Modell ausgewählt."

                # Lade Tokenizer
                tokenizer = BPETokenizer()
                tokenizer.load(os.path.join("runs", run_name, "tokenizer", "bpe_merges.txt"))
                tokenizer.load_vocab(os.path.join("runs", run_name, "tokenizer", "bpe_vocab.txt"))

                with open(os.path.join("runs", run_name, "config.json")) as f:
                    cfg = json.load(f)

                config = GPTConfig(**cfg)
                model = GPTModel(config)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)

                checkpoints = sorted(
                    [f for f in os.listdir(os.path.join("runs", run_name)) if f.startswith("gpt_epoch") and f.endswith(".pt")],
                    key=lambda x: int(x.split("epoch")[-1].split(".")[0])
                )
                if not checkpoints:
                    return "❌ Kein Modell-Checkpoint gefunden."
                ckpt_path = os.path.join("runs", run_name, checkpoints[-1])
                load_model(model, ckpt_path, device)

                if prompt.strip():
                    start_ids = tokenizer.encode(prompt)
                else:
                    start_ids = [tokenizer.token2id.get("<s>", 0)]

                sample_ids = model.generate(
                    start_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None
                )

                return tokenizer.decode(sample_ids)

            refresh_gen_btn.click(list_runs, outputs=selected_gen_run)

            gen_btn.click(
                generate_text_from_model,
                inputs=[selected_gen_run, prompt_input, max_tokens, temperature, top_k],
                outputs=gen_output
            )


    

if __name__ == "__main__":
    demo.launch()
