from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]

# Legacy flat paths (kept for backwards-compatibility imports elsewhere)
CHECKPOINT_ROOT = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"

PARAPHRASE_LABEL_MAP = {
    0: "Not Paraphrase",
    1: "Paraphrase",
}

# ---------------------------------------------------------------------------
# Variant support: "base" (pre-trained, no fine-tuning) vs "tuned"
# ---------------------------------------------------------------------------

Variant = Literal["base", "tuned"]
VARIANTS: tuple[Variant, ...] = ("base", "tuned")

DATASETS = ("mrpc", "stsb", "qqp")

# Models that appear in each variant
BASE_MODELS = ["Siamese-LSTM", "Siamese-GRU", "BERT", "RoBERTa", "DistilBERT"]
TUNED_MODELS = ["Siamese-LSTM", "Siamese-GRU", "BERT", "RoBERTa", "DistilBERT"]
TUNED_STS_EXTRA = ["SBERT"]  # SBERT only in tuned STS-B


def checkpoint_root_for(variant: Variant) -> Path:
    """Return the checkpoint root directory for a given variant."""
    return ROOT_DIR / "checkpoints" / variant


def results_dir_for(variant: Variant) -> Path:
    """Return the results directory for a given variant."""
    return ROOT_DIR / "results" / variant


def plots_dir_for(variant: Variant) -> Path:
    """Return the plots directory for a given variant."""
    return ROOT_DIR / "plots" / variant


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
