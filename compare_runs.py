# compare_runs.py
import os
import json
import pandas as pd

def load_run_info(run_path):
    config_path = os.path.join(run_path, "config.json")
    log_path = os.path.join(run_path, "training_log.json")

    if not os.path.exists(config_path) or not os.path.exists(log_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        return None

    final = log[-1]
    return {
        "run_name": os.path.basename(run_path),
        "epoch": final["epoch"],
        "train_loss": final["train_loss"],
        "val_loss": final["val_loss"],
        "perplexity": final["val_perplexity"],
        "n_layer": config.get("n_layer"),
        "n_embd": config.get("n_embd"),
        "block_size": config.get("block_size"),
        "dropout": config.get("dropout"),
        "vocab_size": config.get("vocab_size"),
    }

def collect_all_runs(runs_root="runs"):
    runs = []
    for run_name in os.listdir(runs_root):
        run_path = os.path.join(runs_root, run_name)
        if os.path.isdir(run_path):
            info = load_run_info(run_path)
            if info:
                runs.append(info)
    return pd.DataFrame(runs)

if __name__ == "__main__":
    df = collect_all_runs()
    if df.empty:
        print("No valid runs found.")
    else:
        df = df.sort_values(by="perplexity")
        print("\n=== All Completed Runs Sorted by Perplexity ===\n")
        print(df.to_string(index=False))