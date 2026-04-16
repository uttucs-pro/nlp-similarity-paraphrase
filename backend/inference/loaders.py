from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    from src.siamese_model import SiameseGRU, SiameseLSTM
    from src.glove_utils import load_glove_embeddings, build_vocab, DEFAULT_GLOVE_PATH
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.siamese_model import SiameseGRU, SiameseLSTM
    from backend.src.glove_utils import load_glove_embeddings, build_vocab, DEFAULT_GLOVE_PATH


SIAMESE_MODEL_TYPES = {
    "Siamese-LSTM": SiameseLSTM,
    "Siamese-GRU": SiameseGRU,
}

BASE_TRANSFORMER_IDS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}


def resolve_runtime_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_artifact_path(root: Path, value: str | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


# ---------------------------------------------------------------------------
# Tuned model loaders (fine-tuned checkpoints)
# ---------------------------------------------------------------------------

def load_transformer_bundle(manifest: dict, checkpoint_root: Path, device: torch.device) -> dict:
    model_path = resolve_artifact_path(checkpoint_root, manifest["checkpoint_path"])
    tokenizer_path = resolve_artifact_path(
        checkpoint_root,
        manifest["tokenizer_or_vocab_path"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    model.eval()
    return {
        "manifest": manifest,
        "model": model,
        "tokenizer": tokenizer,
    }


def load_siamese_bundle(manifest: dict, checkpoint_root: Path, device: torch.device) -> dict:
    artifact_path = resolve_artifact_path(checkpoint_root, manifest["checkpoint_path"])
    vocab_path = resolve_artifact_path(checkpoint_root, manifest["tokenizer_or_vocab_path"])

    artifact = torch.load(artifact_path, map_location=device)
    with vocab_path.open("r", encoding="utf-8") as handle:
        word2idx = json.load(handle)

    model_cls = SIAMESE_MODEL_TYPES[manifest["model_name"]]
    embedding_matrix = np.array(artifact["embedding_matrix"], dtype=np.float32)
    model = model_cls(
        embedding_matrix,
        hidden_dim=artifact["hidden_dim"],
        num_layers=artifact["num_layers"],
        dropout=artifact["dropout"],
        task=artifact["task"],
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    return {
        "manifest": manifest,
        "model": model,
        "word2idx": word2idx,
        "max_len": artifact["max_len"],
        "device": device,
    }


def load_sbert_bundle(manifest: dict, device: torch.device) -> dict:
    model_id = manifest["model_id"]
    sbert = SentenceTransformer(model_id, device=str(device))
    return {
        "manifest": manifest,
        "model": sbert,
    }


# ---------------------------------------------------------------------------
# Base model loaders (pre-trained, no fine-tuning)
# ---------------------------------------------------------------------------

def load_base_transformer_bundle(manifest: dict, device: torch.device) -> dict:
    """Load a pre-trained transformer encoder (AutoModel) for embedding-based inference."""
    model_id = manifest.get("model_id") or BASE_TRANSFORMER_IDS.get(manifest["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return {
        "manifest": manifest,
        "model": model,
        "tokenizer": tokenizer,
        "model_id": model_id,
    }


def load_base_glove_bundle(manifest: dict, device: torch.device) -> dict:
    """
    Load GloVe embeddings for base Siamese models (mean-pooling inference).

    Since base Siamese models have no trained encoder, we just need
    the GloVe embedding matrix and vocabulary for mean-pooling.
    """
    glove_path = manifest.get("glove_path", str(DEFAULT_GLOVE_PATH))

    # Build a minimal vocab from the GloVe file itself
    # For serving, we load all GloVe words
    word2idx = {}
    embedding_list = []
    embed_dim = 100

    # Pad and UNK
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    embedding_list.append(np.zeros(embed_dim, dtype=np.float32))  # PAD
    embedding_list.append(np.random.uniform(-0.25, 0.25, embed_dim).astype(np.float32))  # UNK

    glove_file = Path(glove_path)
    if glove_file.exists():
        idx = 2
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word not in word2idx:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == embed_dim:
                        word2idx[word] = idx
                        embedding_list.append(vector)
                        idx += 1

    embedding_matrix = np.array(embedding_list, dtype=np.float32)

    return {
        "manifest": manifest,
        "word2idx": word2idx,
        "embedding_matrix": embedding_matrix,
    }
