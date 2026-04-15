from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

PARAPHRASE_LABEL_MAP = {
    0: "Not Paraphrase",
    1: "Paraphrase",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def resolve_training_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
