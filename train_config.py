import os
from dataclasses import dataclass
from typing import Dict, Any
import torch
import json


# ------------------------------
# Config
# ------------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    use_amp: bool = True
    grad_accum_steps: int = 1  #
    # max_grad_norm: float = 1.0
    early_stop_patience: int = 5
    early_stop_tolerance: float = 0  # tolerance for early stopping
    seed: int = 666
    ckpt_n_epochs: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt_best_filename: str = "best.pt"
    ckpt_last_filename: str = "last.pt"
    log_dir: str = "logs"
    ckpt_best_path: str = None
    ckpt_last_path: str = None

    def __post_init__(self):
        # ensure checkpoint directory exists
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_last_path = os.path.join(self.ckpt_dir, self.ckpt_last_filename)
        self.ckpt_best_path = os.path.join(self.ckpt_dir, self.ckpt_best_filename)
        os.makedirs(self.log_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "epochs": self.epochs,
            "use_amp": self.use_amp,
            "grad_accum_steps": self.grad_accum_steps,
            # "max_grad_norm": self.max_grad_norm,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_tolerance": self.early_stop_tolerance,
            "seed": self.seed,
            "ckpt_n_epochs": self.ckpt_n_epochs,
            "ckpt_dir": self.ckpt_dir,
            "ckpt_best_filename": self.ckpt_best_filename,
            "ckpt_last_filename": self.ckpt_last_filename,
            "log_dir": self.log_dir,
            "ckpt_best_path": self.ckpt_best_path,
            "ckpt_last_path": self.ckpt_last_path,
        }

    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        return cls(
            device=data.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            epochs=data.get("epochs", 10),
            use_amp=data.get("use_amp", True),
            grad_accum_steps=data.get("grad_accum_steps", 1),
            # max_grad_norm=data.get("max_grad_norm", 1.0),
            early_stop_patience=data.get("early_stop_patience", 5),
            early_stop_tolerance=data.get("early_stop_tolerance", 0.0),
            seed=data.get("seed", 666),
            ckpt_n_epochs=data.get("ckpt_n_epochs", 5),
            ckpt_dir=data.get("ckpt_dir", "checkpoints"),
            ckpt_best_filename=data.get("ckpt_best_filename", "best.pt"),
            ckpt_last_filename=data.get("ckpt_last_filename", "last.pt"),
            log_dir=data.get("log_dir", "logs"),
            ckpt_best_path=data.get("ckpt_best_path", "checkpoints/best.pt"),
            ckpt_last_path=data.get("ckpt_last_path", "checkpoints/last.pt"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TrainConfig":
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        return f"TrainConfig({self.to_dict()})"

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
