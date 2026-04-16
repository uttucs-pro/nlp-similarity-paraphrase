from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry: dataset → variant → model names
# ---------------------------------------------------------------------------

DATASETS = {
    "mrpc": {
        "task": "paraphrase_detection",
        "models": ["Siamese-LSTM", "Siamese-GRU", "BERT", "RoBERTa", "DistilBERT"],
    },
    "qqp": {
        "task": "paraphrase_detection",
        "models": ["Siamese-LSTM", "Siamese-GRU", "BERT", "RoBERTa", "DistilBERT"],
    },
    "stsb": {
        "task": "semantic_similarity",
        "models": ["Siamese-LSTM", "Siamese-GRU", "BERT", "RoBERTa", "DistilBERT"],
    },
}

# SBERT only in tuned STS-B
TUNED_STS_EXTRA = ["SBERT"]

VARIANTS = ("base", "tuned")


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "-")


def manifest_filename(dataset: str, variant: str, model_name: str) -> str:
    """Generate manifest filename: {dataset}__{variant}__{model-slug}.json"""
    return f"{dataset}__{variant}__{slugify_model_name(model_name)}.json"


def get_models_for(dataset: str, variant: str) -> list[str]:
    """Return the list of model names for a given dataset and variant."""
    models = list(DATASETS[dataset]["models"])
    if variant == "tuned" and dataset == "stsb":
        models.extend(TUNED_STS_EXTRA)
    return models


def get_task_for(dataset: str) -> str:
    """Return the task type for a dataset."""
    return DATASETS[dataset]["task"]


def load_required_manifests(
    checkpoint_root: Path,
    strict: bool = True,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Load all required manifests organized by dataset → variant → model_name.

    For base models, manifests are under checkpoint_root/base/manifests/
    For tuned models, manifests are under checkpoint_root/tuned/manifests/

    Returns:
        {
            "mrpc": {
                "base":  {"Siamese-LSTM": {...}, ...},
                "tuned": {"Siamese-LSTM": {...}, ...}
            },
            ...
        }
    """
    manifests: dict[str, dict[str, dict[str, dict]]] = {}

    for dataset in DATASETS:
        manifests[dataset] = {}
        for variant in VARIANTS:
            manifests[dataset][variant] = {}
            manifest_root = checkpoint_root / variant / "manifests"
            model_names = get_models_for(dataset, variant)

            for model_name in model_names:
                fname = manifest_filename(dataset, variant, model_name)
                manifest_path = manifest_root / fname
                if not manifest_path.exists():
                    if strict:
                        raise FileNotFoundError(
                            f"Required manifest not found: {manifest_path}"
                        )
                    continue
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                manifests[dataset][variant][model_name] = manifest

    return manifests
