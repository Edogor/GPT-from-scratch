import os
import json
from datetime import datetime
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
import io

import gradio as gr
import torch
import pandas as pd

from gpt_model import GPTConfig, GPTModel
from train_utils import load_dataset, train_model, load_model, plot_training_log
from bpe_tokenizer import BPETokenizer

# ------------------------------------------------------------
# Pfade
# ------------------------------------------------------------
TRAIN_PATH = "../corpora/Shakespeare_clean_train.txt"
VAL_PATH   = "../corpora/Shakespeare_clean_valid.txt"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _sorted_checkpoints(run_dir: str):
    """Sortiere gpt_epoch{N}.pt numerisch aufsteigend."""
    if not os.path.isdir(run_dir):
        return []
    cand = [f for f in os.listdir(run_dir) if f.startswith("gpt_epoch") and f.endswith(".pt")]
    def _key(x):
        try:
            return int(x.split("epoch")[-1].split(".")[0])
        except Exception:
            return -1
    return sorted(cand, key=_key)

def _list_runs_dir():
    return sorted(os.listdir("runs")) if os.path.isdir("runs") else []

def _safe_plot(log_path: str, run_dir: str):
    """
    Erzeuge/aktualisiere runs/<run>/curve.png stabil und leise.
    - Aufruf von plot_training_log NUR wenn Log Einträge hat
    - stdout/stderr aus dem Plotter werden geschluckt (kein "No log data found." Spam)
    - Fallback: Matplotlib aus JSON
    """
    out_png = os.path.abspath(os.path.join(run_dir, "curve.png"))

    # Vorab: Log prüfen – wenn leer/fehlt, NICHT plotten
    try:
        if not os.path.exists(log_path):
            return None
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        if not isinstance(log, list) or len(log) == 0:
            return None
    except Exception:
        return None

    # 1) Dein Plotter (var. A/B) – Prints unterdrücken
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                p = plot_training_log(log_path, out_dir=run_dir)  # häufige Signatur
            except TypeError:
                p = plot_training_log(log_path)  # alternative Signatur
        if isinstance(p, str) and os.path.exists(p):
            if os.path.abspath(p) != out_png:
                import shutil
                os.makedirs(run_dir, exist_ok=True)
                shutil.copyfile(p, out_png)
            return out_png if os.path.exists(out_png) else os.path.abspath(p)
    except Exception:
        pass

    # 2) Fallback: Matplotlib direkt aus JSON
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r.get("epoch", None) for r in log if "epoch" in r]
        tr = [r.get("train_loss", None) for r in log]
        vl = [r.get("val_loss", None) for r in log]

        os.makedirs(run_dir, exist_ok=True)
        plt.figure()
        if all(e is not None for e in epochs) and len(epochs) == len(tr) == len(vl):
            plt.plot(epochs, tr, label="train")
            plt.plot(epochs, vl, label="val")
            plt.xlabel("epoch")
        else:
            plt.plot(tr, label="train")
            plt.plot(vl, label="val")
            plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend(loc="best")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        return out_png if os.path.exists(out_png) else None
    except Exception:
        return None

# ------------------------------------------------------------
# Streaming-Training (Option B: alles im Thread, UI pollt Mailbox + Log)
# ------------------------------------------------------------
def start_training_stream(k_merges, n_layer, n_embd, n_head, block_size, dropout):
    n_head = int(n_head)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{timestamp}_k{k_merges}_L{n_layer}_E{n_embd}"
    run_dir  = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "tokenizer"), exist_ok=True)

    # UI-Streaming-Mailbox
    last_payload = {"value": None}
    stop_flag    = {"done": False, "error": None}

    # Erste Meldung sofort zeigen
    yield (f"📦 Creating run: {run_name}", gr.update(value=None), "—", 0.0)

    def _emit_from_log(log_path: str, run_dir: str):
        last_seen_epoch = None
        last_plot_mtime = None

        while not stop_flag["done"]:
            try:
                img_upd = gr.update()
                newest_epoch_from_log = None
                status = "⏳ Waiting for log…"
                log = []

                if os.path.exists(log_path):
                    mtime = os.path.getmtime(log_path)
                    with open(log_path, "r", encoding="utf-8") as f:
                        log = json.load(f)

                    if isinstance(log, list) and len(log) > 0:
                        rec = log[-1]
                        newest_epoch_from_log = rec.get("epoch", None)
                        train_loss = rec.get("train_loss", float("nan"))
                        val_loss   = rec.get("val_loss", float("nan"))
                        ppl        = rec.get("val_perplexity", float("nan"))
                        status = f"Epoch {newest_epoch_from_log} | train {train_loss:.4f} | val {val_loss:.4f} | ppl {ppl:.2f}"

                        # Plot nur bei Änderung + wenn Daten vorhanden
                        if last_plot_mtime is None or mtime > last_plot_mtime:
                            img_path = _safe_plot(log_path, run_dir)
                            if img_path:
                                img_upd = gr.update(value=os.path.abspath(img_path))
                            last_plot_mtime = mtime

                # Fallback über neueste Checkpoint-Epoche
                cps = _sorted_checkpoints(run_dir)
                newest_epoch_from_ckpt = None
                if cps:
                    try:
                        newest_epoch_from_ckpt = int(cps[-1].split("epoch")[-1].split(".")[0])
                    except Exception:
                        pass

                display_epoch = newest_epoch_from_log
                if newest_epoch_from_ckpt is not None:
                    if (display_epoch is None) or (newest_epoch_from_ckpt > display_epoch):
                        display_epoch = newest_epoch_from_ckpt
                        if not log or newest_epoch_from_ckpt > (newest_epoch_from_log or -1):
                            status = f"Epoch {display_epoch} | (from ckpt) — waiting for log flush…"

                # Progress/Substatus
                if display_epoch != last_seen_epoch and display_epoch is not None:
                    sub, prog = f"Epoch {display_epoch} (summary)", 1.0
                    last_seen_epoch = display_epoch
                else:
                    sub, prog = f"Epoch {display_epoch if display_epoch is not None else '?'} (running)", 0.0

                last_payload["value"] = (status, img_upd, sub, prog)

            except Exception:
                pass

            time.sleep(0.5)


    def _train_thread():
        """Komplette Pipeline (Prep + train_model) ohne train_utils-Modifikation."""
        try:
            # PREP: Dataset
            last_payload["value"] = ("📚 Loading dataset…", gr.update(), "Preparing data", 0.05)
            train_text, val_text = load_dataset(TRAIN_PATH, VAL_PATH)

            # PREP: Tokenizer
            last_payload["value"] = ("🔤 Training BPE tokenizer…", gr.update(), f"num_merges = {k_merges}", 0.10)
            tokenizer = BPETokenizer(num_merges=int(k_merges))
            tokenizer.train(train_text)

            tok_dir = os.path.join(run_dir, "tokenizer")
            tokenizer.save(os.path.join(tok_dir, "bpe_merges.txt"))
            tokenizer.save_vocab(os.path.join(tok_dir, "bpe_vocab.txt"))

            # PREP: Encoding
            last_payload["value"] = ("✍️ Encoding train set…", gr.update(), "Encoding tokens (train)", 0.20)
            train_ids = tokenizer.encode(train_text)
            last_payload["value"] = ("✍️ Encoding val set…", gr.update(), "Encoding tokens (val)", 0.25)
            val_ids   = tokenizer.encode(val_text)

            # PREP: Model init
            vocab_size = len(tokenizer.vocab)
            config = GPTConfig(
                vocab_size=vocab_size,
                block_size=int(block_size),
                n_layer=int(n_layer),
                n_head=int(n_head),
                n_embd=int(n_embd),
                dropout=float(dropout),
            )
            model  = GPTModel(config)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            last_payload["value"] = ("🧠 Model initialized", gr.update(), f"device = {device}", 0.30)

            # Dateien / Resume
            checkpoint_base = os.path.join(run_dir, "gpt_epoch")
            log_path        = os.path.join(run_dir, "training_log.json")
            config_path     = os.path.join(run_dir, "config.json")

            # <<< NEU: leere Log einmalig anlegen, aber Plotter wird erst bei Einträgen aufgerufen
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("[]")

            cps = _sorted_checkpoints(run_dir)
            resume_from = os.path.join(run_dir, cps[-1]) if cps else None

            # Watcher starten (für Plot + Anzeige)
            watcher = threading.Thread(target=_emit_from_log, args=(log_path, run_dir), daemon=True)
            watcher.start()

            # TRAIN
            last_payload["value"] = ("🚀 Starting training…", gr.update(), "Epoch 1", 0.35)
            train_model(
                model, train_ids, val_ids, config, device,
                checkpoint_base=checkpoint_base,
                log_path=log_path,
                config_path=config_path,
                resume_from=resume_from,
                max_batches_per_epoch=1000,
            )

            stop_flag["done"] = True

        except Exception as e:
            stop_flag["error"] = str(e)
            stop_flag["done"]  = True

    # Threads starten
    th = threading.Thread(target=_train_thread, daemon=True)
    th.start()

    # UI stream loop
    while not stop_flag["done"]:
        if last_payload["value"] is not None:
            s, img_update, sub, prog = last_payload["value"]
            yield (s, img_update, sub, prog)
            last_payload["value"] = None
        time.sleep(0.2)

    # final flush
    if last_payload["value"] is not None:
        s, img_update, sub, prog = last_payload["value"]
        yield (s, img_update, sub, prog)
        last_payload["value"] = None

    if stop_flag["error"]:
        yield (f"❌ Training failed: {stop_flag['error']}", gr.update(), "Error", 0.0)
    else:
        yield (f"🎉 Training finished: {run_name}", gr.update(), "Done", 1.0)

# ------------------------------------------------------------
# Run-Übersicht
# ------------------------------------------------------------
def get_run_info():
    runs = []
    if not os.path.isdir("runs"):
        return pd.DataFrame(runs)
    for run in sorted(os.listdir("runs")):
        run_dir = os.path.join("runs", run)
        try:
            with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(os.path.join(run_dir, "training_log.json"), "r", encoding="utf-8") as f:
                log = json.load(f)
            if not log:
                continue
            last = log[-1]
            runs.append({
                "run": run,
                "epoch": last.get("epoch"),
                "train_loss": round(last.get("train_loss", float('nan')), 3),
                "val_loss": round(last.get("val_loss", float('nan')), 3),
                "ppl": round(last.get("val_perplexity", float('nan')), 2),
                "n_layer": cfg.get("n_layer"),
                "n_embd": cfg.get("n_embd"),
                "block_size": cfg.get("block_size"),
            })
        except Exception:
            continue
    return pd.DataFrame(runs)

# ------------------------------------------------------------
# Textgenerierung
# ------------------------------------------------------------
def list_runs_dropdown():
    return gr.update(choices=_list_runs_dir())

def generate_text_from_model(run_name, prompt, max_tokens, temperature, top_k):
    if not run_name:
        return "❌ Kein Modell ausgewählt."
    run_dir = os.path.join("runs", run_name)
    tok_dir = os.path.join(run_dir, "tokenizer")

    # Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(tok_dir, "bpe_merges.txt"))
    tokenizer.load_vocab(os.path.join(tok_dir, "bpe_vocab.txt"))

    # Config + Modell
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    config = GPTConfig(**cfg)
    model  = GPTModel(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cps = _sorted_checkpoints(run_dir)
    if not cps:
        return "❌ Kein Modell-Checkpoint gefunden."
    ckpt_path = os.path.join(run_dir, cps[-1])
    load_model(model, ckpt_path, device)

    # Start-IDs
    if prompt and prompt.strip():
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

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
with gr.Blocks() as demo:
    with gr.Tab("📦 Neues Modell erstellen & trainieren"):
        gr.Markdown("### GPT-Modell starten")

        k_merges   = gr.Slider(0, 1000, value=200, step=50, label="BPE Merges (k)")
        n_layer    = gr.Slider(1, 12, value=4, step=1, label="Anzahl Layer")
        n_embd     = gr.Slider(2, 512, value=128, step=32, label="Embedding-Dimension")
        n_head     = gr.Dropdown(choices=["1", "2", "4", "8"], value="4", label="Anzahl Attention-Heads")
        block_size = gr.Slider(16, 256, value=64, step=32, label="Blockgröße (Kontext)")
        dropout    = gr.Slider(0.0, 0.3, value=0.1, step=0.05, label="Dropout")

        start_btn = gr.Button("🚀 Training starten", variant="primary")

        status    = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            live_plot = gr.Image(label="Training Curve (updates live)", interactive=False)
            with gr.Column():
                substatus = gr.Textbox(label="Aktueller Schritt", interactive=False)
                progress  = gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Epoch-Progress", interactive=False)

        # Heads-Validierung
        def update_heads(n_embd_value):
            n_embd_value = int(n_embd_value)
            valid_heads = [str(h) for h in range(1, n_embd_value + 1) if n_embd_value % h == 0]
            default = "4" if "4" in valid_heads else valid_heads[0]
            return gr.update(choices=valid_heads, value=default)

        n_embd.change(update_heads, inputs=n_embd, outputs=n_head)

        start_btn.click(
            start_training_stream,
            inputs=[k_merges, n_layer, n_embd, n_head, block_size, dropout],
            outputs=[status, live_plot, substatus, progress]
        )

    with gr.Tab("📂 Vorhandene Runs"):
        gr.Markdown("### Wähle ein vorhandenes Modell")
        df = gr.Dataframe(
            headers=["run", "epoch", "train_loss", "val_loss", "ppl", "n_layer", "n_embd", "block_size"],
            interactive=False,
            label="Modellübersicht"
        )
        refresh_btn  = gr.Button("🔄 Liste aktualisieren")
        selected_txt = gr.Textbox(label="Gewählter Run (zum Kopieren)", interactive=False)

        def refresh_table():
            table = get_run_info()
            return table.values.tolist() if not table.empty else []

        def select_row(evt: gr.SelectData):
            row_idx = evt.index[0]
            table = get_run_info()
            try:
                return table.iloc[row_idx]["run"]
            except Exception:
                return ""

        refresh_btn.click(refresh_table, outputs=df)
        df.select(select_row, outputs=selected_txt)

        with gr.Tab("🧠 Text generieren"):
            gr.Markdown("### Wähle ein Modell und generiere Text")

            with gr.Row():
                selected_gen_run = gr.Dropdown(
                    choices=_list_runs_dir(),
                    label="Modell wählen (Run-Ordner)"
                )
                refresh_gen_btn = gr.Button("🔄")
                refresh_gen_btn.click(lambda: gr.update(choices=_list_runs_dir()), outputs=selected_gen_run)

            prompt_input = gr.Textbox(label="Prompt (optional)", placeholder="z. B. Once upon a time")
            max_tokens   = gr.Slider(10, 500, value=100, step=10, label="Max neue Tokens")
            temperature  = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k        = gr.Slider(0, 100, value=20, step=1, label="Top-k (0 = deaktiviert)")

            gen_btn    = gr.Button("🎬 Text generieren")
            gen_output = gr.Textbox(label="Output")

            gen_btn.click(
                generate_text_from_model,
                inputs=[selected_gen_run, prompt_input, max_tokens, temperature, top_k],
                outputs=gen_output
            )

if __name__ == "__main__":
    demo.launch()
