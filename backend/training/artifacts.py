from __future__ import annotations

from pathlib import Path

import torch

try:
    from inference.registry import manifest_filename
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.inference.registry import manifest_filename

from .common import ensure_dir, write_json


def _relative_to_root(path: Path, checkpoint_root: Path) -> str:
    return path.resolve().relative_to(checkpoint_root.resolve()).as_posix()


# ---------------------------------------------------------------------------
# Base (pre-trained, no fine-tuning) manifest export
# ---------------------------------------------------------------------------

def export_base_manifest(
    *,
    dataset: str,
    model_name: str,
    family: str,
    model_id: str | None = None,
    glove_path: str | None = None,
    manifest_root: Path,
) -> dict:
    """
    Export a manifest for a base (pre-trained, no fine-tuning) model.

    Base models have no custom checkpoints — they reference pre-trained
    model IDs (for transformers) or GloVe paths (for Siamese).
    """
    task = "semantic_similarity" if dataset == "stsb" else "paraphrase_detection"

    manifest = {
        "task": task,
        "dataset": dataset,
        "variant": "base",
        "model_name": model_name,
        "family": family,
        "checkpoint_path": "",
        "tokenizer_or_vocab_path": "",
        "max_len": 128,
        "output_type": "regression" if dataset == "stsb" else "classification",
        "scale": "0-1",
        "label_map": {"0": "Not Paraphrase", "1": "Paraphrase"} if task == "paraphrase_detection" else {},
        "preprocessing_mode": "normalise_sentence",
        "inference_mode": "embedding_similarity",
    }

    if model_id:
        manifest["model_id"] = model_id
    if glove_path:
        manifest["glove_path"] = glove_path

    write_json(manifest_root / manifest_filename(dataset, "base", model_name), manifest)
    return manifest


# ---------------------------------------------------------------------------
# Tuned (fine-tuned) checkpoint export — Siamese
# ---------------------------------------------------------------------------

def export_siamese_checkpoint(
    *,
    task: str,
    dataset: str,
    model_name: str,
    model,
    output_dir: Path,
    embedding_matrix,
    word2idx: dict[str, int],
    max_len: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    label_map: dict[int, str] | None,
    scale: str,
    checkpoint_root: Path,
    manifest_root: Path,
) -> dict:
    ensure_dir(output_dir)

    weights_path = output_dir / "model_state.pt"
    vocab_path = output_dir / "vocab.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_matrix": embedding_matrix.tolist(),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "task": "classification" if task == "paraphrase_detection" else "regression",
            "max_len": max_len,
        },
        weights_path,
    )
    write_json(vocab_path, word2idx)

    manifest = {
        "task": task,
        "dataset": dataset,
        "variant": "tuned",
        "model_name": model_name,
        "family": "siamese",
        "checkpoint_path": _relative_to_root(weights_path, checkpoint_root),
        "tokenizer_or_vocab_path": _relative_to_root(vocab_path, checkpoint_root),
        "max_len": max_len,
        "output_type": "classification" if task == "paraphrase_detection" else "regression",
        "scale": scale,
        "label_map": {str(key): value for key, value in (label_map or {}).items()},
        "preprocessing_mode": "normalise_sentence",
        "inference_mode": "model_forward",
    }
    write_json(manifest_root / manifest_filename(dataset, "tuned", model_name), manifest)
    return manifest


# ---------------------------------------------------------------------------
# Tuned (fine-tuned) checkpoint export — Transformer
# ---------------------------------------------------------------------------

def export_transformer_checkpoint(
    *,
    task: str,
    dataset: str,
    model_name: str,
    model,
    tokenizer,
    output_dir: Path,
    max_len: int,
    label_map: dict[int, str] | None,
    scale: str,
    checkpoint_root: Path,
    manifest_root: Path,
) -> dict:
    model_dir = ensure_dir(output_dir / "model")
    tokenizer_dir = ensure_dir(output_dir / "tokenizer")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    task_str = task
    manifest = {
        "task": task_str,
        "dataset": dataset,
        "variant": "tuned",
        "model_name": model_name,
        "family": "transformer",
        "checkpoint_path": _relative_to_root(model_dir, checkpoint_root),
        "tokenizer_or_vocab_path": _relative_to_root(tokenizer_dir, checkpoint_root),
        "max_len": max_len,
        "output_type": "classification" if task == "paraphrase_detection" else "regression",
        "scale": scale,
        "label_map": {str(key): value for key, value in (label_map or {}).items()},
        "preprocessing_mode": "normalise_sentence",
        "inference_mode": "model_forward",
    }
    write_json(manifest_root / manifest_filename(dataset, "tuned", model_name), manifest)
    return manifest


# ---------------------------------------------------------------------------
# SBERT manifest (zero-shot, only in tuned STS-B)
# ---------------------------------------------------------------------------

def export_sbert_manifest(
    *,
    model_name: str,
    model_id: str,
    manifest_root: Path,
) -> dict:
    manifest = {
        "task": "semantic_similarity",
        "dataset": "stsb",
        "variant": "tuned",
        "model_name": model_name,
        "family": "sbert",
        "checkpoint_path": "",
        "tokenizer_or_vocab_path": "",
        "max_len": 0,
        "output_type": "regression",
        "scale": "0-1",
        "label_map": {},
        "preprocessing_mode": "normalise_sentence",
        "model_id": model_id,
        "inference_mode": "model_forward",
    }
    write_json(
        manifest_root / manifest_filename("stsb", "tuned", model_name),
        manifest,
    )
    return manifest
