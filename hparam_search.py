import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import os, json, time, uuid
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List
from itertools import product

from train_mj import train, ConfigTrain
from neural_bigram import NeuralBigram, ConfigNeuralBigram
from GPT_mj import GPT, ConfigGPT
from utils import WarmupThenCosine, init_dataloader, count_params
from bpe_hf import train_and_encode_tokenizer, train_bytelevel_bpe, SPECIAL_TOKENS


@dataclass
class hparamsSpace:
    """
    abstract base class for hparam search spaces
    all fields must be non-empty iterables
    """

    def __post_init__(self):
        # all showuld be iterables
        assert all(
            hasattr(v, "__iter__") or hasattr(v, "__getitem__") for k, v in self.__dict__.items()
        ), "All hparam space values must be iterables"
        # no empty iterables
        assert all(len(v) > 0 for k, v in self.__dict__.items()), "All hparam space iterables must be non-empty"

    def num_total_combinations(self) -> int:
        return np.prod([len(v) for k, v in self.__dict__.items()])


@dataclass
class hparamsSpaceNBigram(hparamsSpace):
    merges: Iterable[int] = (200,)
    dropout: Iterable[float] = (0.2,)
    lr: Iterable[float] = (3e-3,)
    weight_decay: Iterable[float] = (1e-4,)
    lr_scheduler: Iterable[str] = ("cosine_warmup",)  # ("cosine", "cosine_restarts", cosine_warmup, ...)


@dataclass
class hparamsSpaceGPT(hparamsSpace):
    merges: Iterable[int] = (200,)
    n_embed: Iterable[int] = (64,)
    n_heads: Iterable[int] = (4,)
    n_layers: Iterable[int] = (4,)
    dropout: Iterable[float] = (0.2,)
    lr: Iterable[float] = (3e-3,)
    weight_decay: Iterable[float] = (1e-4,)
    lr_scheduler: Iterable[str] = ("cosine_warmup",)


def _trial_run_name(base: str = "sweep") -> str:
    return f"{base}_{time.strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _safe_lr_sched(name: str, optimizer, total_steps: int, eta_min: float):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    if name == "cosine_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=max(2, total_steps // 10), T_mult=2, eta_min=eta_min)
    if name == "cosine_warmup":
        return WarmupThenCosine(optimizer, warmup_steps=500, T_max=total_steps, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {name}")


# region nbigram hparam search
def hparams_search_nBigram(
    *,
    # search space
    hp_space: hparamsSpaceNBigram,
    # training
    base_cfg_train: ConfigTrain,
    base_cfg_model: ConfigNeuralBigram,
    train_text_path: str,
    val_text_path: str,
    # tokenizer
    tokenizer_trainer: callable = train_bytelevel_bpe,  # train_bytelevel_bpe
    special_tokens: dict = SPECIAL_TOKENS,
    tok_min_frequency: int = 2,
    # data
    batch_size: int = 32,
    block_size: int = 128,
    # lr scheduler
    eta_min: float = 1e-8,
    verbose: bool = False,
):
    os.makedirs(base_cfg_train.log_dir, exist_ok=True)
    os.makedirs(base_cfg_train.ckpt_dir, exist_ok=True)

    total_runs = hp_space.num_total_combinations()

    rows: List[Dict[str, Any]] = []
    pbar_all = tqdm(total=total_runs, desc="HParams Search", leave=True, unit="trial")

    for merges in hp_space.merges:
        tok_info = train_and_encode_tokenizer(
            tokenizer_trainer=tokenizer_trainer,
            train_text_path=train_text_path,
            other_texts_paths={"val": val_text_path},
            merges=merges,
            min_frequency=tok_min_frequency,
            special_tokens=special_tokens,
        )
        train_ids = tok_info["train_ids"]
        val_ids = tok_info["other_texts_ids"]["val"]
        vocab_sz = tok_info["vocab_size"]

        # dataloaders
        train_loader = init_dataloader(train_ids, block_size, batch_size, train=True, shuffle=True)
        val_loader = init_dataloader(val_ids, block_size, batch_size, train=False, shuffle=True)

        for lr, dropout, weight_decay, sched_name in product(
            hp_space.lr, hp_space.dropout, hp_space.weight_decay, hp_space.lr_scheduler
        ):

            run_name = _trial_run_name("hps")
            run_dir = os.path.join(base_cfg_train.log_dir, run_name)
            writer = SummaryWriter(log_dir=run_dir, flush_secs=5)
            # update model and train cfgs
            cfg_model = type(base_cfg_model)(
                **{
                    **vars(base_cfg_model),
                    "vocab_size": vocab_sz,
                    "dropout": dropout,
                }
            )
            cfg_train = type(base_cfg_train)(
                **{
                    **vars(base_cfg_train),
                    "ckpt_best_filename": f"best_{run_name}.pt",
                    "ckpt_last_filename": f"last_{run_name}.pt",
                    "log_dir": run_dir,
                }
            )

            # build model
            model = NeuralBigram(cfg_model)
            model.to(base_cfg_train.device)
            model.compile(mode="reduce-overhead")
            # optimizer
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # scheduler
            total_steps = base_cfg_train.epochs * max(1, len(train_loader) // max(1, base_cfg_train.grad_accum_steps))
            lr_scheduler = _safe_lr_sched(sched_name, optimizer, total_steps, eta_min)
            # scaler
            scaler = GradScaler(enabled=base_cfg_train.use_amp)

            # train
            out = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_train,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=scaler,
                writer=writer,
                show_pbar=False,
            )

            # log all hparams
            rl_scheduler_state_dict = (
                lr_scheduler.state_dict()
                if lr_scheduler.__class__.__name__ != "WarmupThenCosine"
                else {k: v for k, v in lr_scheduler.state_dict().items() if k != "cosine"}
            )
            hparams_json = {
                "tokenizer": {
                    "type": "bytelevel_bpe",
                    "n_merges": merges,
                    "min_frequency": tok_min_frequency,
                    "special_tokens": special_tokens,
                },
                "data": {
                    "batch_size": batch_size,
                    "block_size": block_size,
                },
                "model": vars(cfg_model).copy(),
                "optimizer": {
                    "type": optimizer.__class__.__name__,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "defaults": optimizer.defaults,
                },
                "lr_scheduler": {
                    "type": lr_scheduler.__class__.__name__,
                    "eta_min": eta_min,
                    "state_dict": rl_scheduler_state_dict,
                },
                "scaler": {
                    "type": scaler.__class__.__name__,
                    "enabled": scaler.is_enabled(),
                    "state_dict": scaler.state_dict(),
                },
                "training": vars(cfg_train).copy(),
            }

            full_config_file = os.path.join(base_cfg_train.ckpt_dir, f"{run_name}_hparams.json")
            with open(full_config_file, "w") as f:
                json.dump(hparams_json, f, indent=4)

            writer.add_text("config/json", "```json\n" + json.dumps(hparams_json, indent=2) + "\n```", 0)
            writer.flush()
            writer.close()

            # final metrics per run
            row = {
                "merges": merges,
                "lr": lr,
                "dropout": dropout,
                "vocab_size": vocab_sz,
                "weight_decay": weight_decay,
                "scheduler": sched_name,
                "val_ppl": out["best_val_ppl"],
                "train_ppl": out["history"]["train_ppl"][-1],
                "val_loss": out["history"]["val_loss"][-1],
                "train_loss": out["history"]["train_loss"][-1],
                "epochs": out["last_epoch"] + 1,
                "ckpt_best": os.path.join(cfg_train.ckpt_best_path),
                "ckpt_last": os.path.join(cfg_train.ckpt_last_path),
                "full_config_file": full_config_file,
                "run": run_name,
            }
            rows.append(row)

            if verbose:
                print(
                    f"[{run_name}] k={merges}, lr={lr:.1e}, drop={dropout}, "
                    f"wd={weight_decay}, sched={sched_name} -> val_ppl {row['val_ppl']:.2f}"
                )

            pbar_all.set_postfix(
                val_ppl=f"{row['val_ppl']:.2f}",
                merges=merges,
                lr=f"{lr:.1e}",
                dropout=dropout,
                weight_decay=weight_decay,
                scheduler=sched_name,
            )
            pbar_all.update(1)

    pbar_all.close()
    df = pd.DataFrame(rows).sort_values(["val_ppl", "merges", "lr", "dropout"]).reset_index(drop=True)
    return df


# region GPT hparam search


def hparams_search_GPT(
    *,
    # search space
    hp_space: hparamsSpaceGPT,
    # training
    base_cfg_train: ConfigTrain,
    base_cfg_model: ConfigGPT,
    train_text_path: str,
    val_text_path: str,
    # tokenizer
    tokenizer_trainer: callable = train_bytelevel_bpe,  # train_bytelevel_bpe
    special_tokens: dict = SPECIAL_TOKENS,
    tok_min_frequency: int = 2,
    # data
    batch_size: int = 32,
    block_size: int = 128,
    # lr scheduler
    eta_min: float = 1e-8,
    verbose: bool = False,
):
    os.makedirs(base_cfg_train.log_dir, exist_ok=True)
    os.makedirs(base_cfg_train.ckpt_dir, exist_ok=True)

    total_runs = hp_space.num_total_combinations()

    rows: List[Dict[str, Any]] = []
    pbar_all = tqdm(total=total_runs, desc="HParams Search", leave=True, unit="trial")

    for merges in hp_space.merges:
        tok_info = train_and_encode_tokenizer(
            tokenizer_trainer=tokenizer_trainer,
            train_text_path=train_text_path,
            other_texts_paths={"val": val_text_path},
            merges=merges,
            min_frequency=tok_min_frequency,
            special_tokens=special_tokens,
        )
        train_ids = tok_info["train_ids"]
        val_ids = tok_info["other_texts_ids"]["val"]
        vocab_sz = tok_info["vocab_size"]

        # dataloaders
        train_loader = init_dataloader(train_ids, block_size, batch_size, train=True, shuffle=True)
        val_loader = init_dataloader(val_ids, block_size, batch_size, train=False, shuffle=True)

        for n_embed, n_heads, n_layers, dropout, lr, weight_decay, sched_name in product(
            hp_space.n_embed,
            hp_space.n_heads,
            hp_space.n_layers,
            hp_space.dropout,
            hp_space.lr,
            hp_space.weight_decay,
            hp_space.lr_scheduler,
        ):

            run_name = _trial_run_name("hps")
            run_dir = os.path.join(base_cfg_train.log_dir, run_name)
            writer = SummaryWriter(log_dir=run_dir, flush_secs=5)
            # update model and train cfgs
            cfg_model = type(base_cfg_model)(
                **{
                    **vars(base_cfg_model),
                    "vocab_size": vocab_sz,
                    "n_embed": n_embed,
                    "n_head": n_heads,
                    "n_layer": n_layers,
                    "dropout": dropout,
                }
            )
            cfg_train = type(base_cfg_train)(
                **{
                    **vars(base_cfg_train),
                    "ckpt_best_filename": f"best_{run_name}.pt",
                    "ckpt_last_filename": f"last_{run_name}.pt",
                    "log_dir": run_dir,
                }
            )
            # build model
            model = GPT(cfg_model)
            model.to(base_cfg_train.device)
            model.compile(mode="reduce-overhead")
            # get model size
            model_size = count_params(model)

            # optimizer
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # scheduler
            total_steps = base_cfg_train.epochs * max(1, len(train_loader) // max(1, base_cfg_train.grad_accum_steps))
            lr_scheduler = _safe_lr_sched(sched_name, optimizer, total_steps, eta_min)
            # scaler
            scaler = GradScaler(enabled=base_cfg_train.use_amp)

            # train
            out = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_train,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=scaler,
                writer=writer,
                show_pbar=False,
            )

            # log all hparams
            rl_scheduler_state_dict = (
                lr_scheduler.state_dict()
                if lr_scheduler.__class__.__name__ != "WarmupThenCosine"
                else {k: v for k, v in lr_scheduler.state_dict().items() if k != "cosine"}
            )
            hparams_json = {
                "tokenizer": {
                    "type": "bytelevel_bpe",
                    "n_merges": merges,
                    "min_frequency": tok_min_frequency,
                    "special_tokens": special_tokens,
                },
                "data": {
                    "batch_size": batch_size,
                    "block_size": block_size,
                },
                "model": vars(cfg_model).copy(),
                "optimizer": {
                    "type": optimizer.__class__.__name__,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "defaults": optimizer.defaults,
                },
                "lr_scheduler": {
                    "type": lr_scheduler.__class__.__name__,
                    "eta_min": eta_min,
                    "state_dict": rl_scheduler_state_dict,
                },
                "scaler": {
                    "type": scaler.__class__.__name__,
                    "enabled": scaler.is_enabled(),
                    "state_dict": scaler.state_dict(),
                },
                "training": vars(cfg_train).copy(),
            }

            full_config_file = os.path.join(base_cfg_train.ckpt_dir, f"{run_name}_hparams.json")
            with open(full_config_file, "w") as f:
                json.dump(hparams_json, f, indent=4)

            writer.add_text("config/json", "```json\n" + json.dumps(hparams_json, indent=2) + "\n```", 0)
            writer.flush()
            writer.close()

            # final metrics per run
            row = {
                "merges": merges,
                "n_embed": n_embed,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "scheduler": sched_name,
                "vocab_size": vocab_sz,
                "model_size": model_size,
                "val_ppl": out["best_val_ppl"],
                "train_ppl": out["history"]["train_ppl"][-1],
                "val_loss": out["history"]["val_loss"][-1],
                "train_loss": out["history"]["train_loss"][-1],
                "epochs": out["last_epoch"] + 1,
                "ckpt_best": os.path.join(cfg_train.ckpt_best_path),
                "ckpt_last": os.path.join(cfg_train.ckpt_last_path),
                "full_config_file": full_config_file,
                "run": run_name,
            }
            rows.append(row)

            if verbose:
                print(
                    f"[{run_name}] k={merges}, lr={lr:.1e}, drop={dropout}, "
                    f"wd={weight_decay}, sched={sched_name} -> val_ppl {row['val_ppl']:.2f}"
                )

            pbar_all.set_postfix(
                val_ppl=f"{row['val_ppl']:.2f}",
                merges=merges,
                lr=f"{lr:.1e}",
                dropout=dropout,
                weight_decay=weight_decay,
                scheduler=sched_name,
            )
            pbar_all.update(1)

    pbar_all.close()
    df = (
        pd.DataFrame(rows)
        .sort_values(["val_ppl", "merges", "n_embed", "n_heads", "n_layers", "lr", "dropout"])
        .reset_index(drop=True)
    )
    return df
