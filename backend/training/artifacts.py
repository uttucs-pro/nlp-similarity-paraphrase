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
        "model_name": model_name,
        "family": "siamese",
        "checkpoint_path": _relative_to_root(weights_path, checkpoint_root),
        "tokenizer_or_vocab_path": _relative_to_root(vocab_path, checkpoint_root),
        "max_len": max_len,
        "output_type": "classification" if task == "paraphrase_detection" else "regression",
        "scale": scale,
        "label_map": {str(key): value for key, value in (label_map or {}).items()},
        "preprocessing_mode": "normalise_sentence",
    }
    write_json(manifest_root / manifest_filename(task, model_name), manifest)
    return manifest


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

    manifest = {
        "task": task,
        "dataset": dataset,
        "model_name": model_name,
        "family": "transformer",
        "checkpoint_path": _relative_to_root(model_dir, checkpoint_root),
        "tokenizer_or_vocab_path": _relative_to_root(tokenizer_dir, checkpoint_root),
        "max_len": max_len,
        "output_type": "classification" if task == "paraphrase_detection" else "regression",
        "scale": scale,
        "label_map": {str(key): value for key, value in (label_map or {}).items()},
        "preprocessing_mode": "normalise_sentence",
    }
    write_json(manifest_root / manifest_filename(task, model_name), manifest)
    return manifest


def export_sbert_manifest(
    *,
    model_name: str,
    model_id: str,
    manifest_root: Path,
) -> dict:
    manifest = {
        "task": "semantic_similarity",
        "dataset": "stsb",
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
    }
    write_json(
        manifest_root / manifest_filename("semantic_similarity", model_name),
        manifest,
    )
    return manifest
