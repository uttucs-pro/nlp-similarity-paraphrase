from __future__ import annotations

import json
from pathlib import Path


REQUIRED_MODELS = {
    "paraphrase_detection": [
        "Siamese-LSTM",
        "Siamese-GRU",
        "BERT",
        "RoBERTa",
        "DistilBERT",
    ],
    "semantic_similarity": [
        "Siamese-LSTM",
        "Siamese-GRU",
        "BERT",
        "RoBERTa",
        "DistilBERT",
        "SBERT",
    ],
}


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "-")


def manifest_filename(task: str, model_name: str) -> str:
    return f"{task}__{slugify_model_name(model_name)}.json"


def load_required_manifests(
    checkpoint_root: Path,
    strict: bool = True,
) -> dict[str, dict[str, dict]]:
    manifest_root = checkpoint_root / "manifests"
    manifests: dict[str, dict[str, dict]] = {}

    for task, model_names in REQUIRED_MODELS.items():
        manifests[task] = {}
        for model_name in model_names:
            manifest_path = manifest_root / manifest_filename(task, model_name)
            if not manifest_path.exists():
                if strict:
                    raise FileNotFoundError(
                        f"Required manifest not found: {manifest_path}"
                    )
                continue
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifests[task][model_name] = manifest

    return manifests
