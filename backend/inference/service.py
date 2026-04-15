from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    from src.preprocessing import normalise_sentence
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.preprocessing import normalise_sentence
from .loaders import (
    load_sbert_bundle,
    load_siamese_bundle,
    load_transformer_bundle,
    resolve_runtime_device,
)
from .registry import load_required_manifests


@dataclass
class PredictionService:
    settings: Any
    device: torch.device = field(init=False)
    ready: bool = field(default=False, init=False)
    models: dict[str, dict[str, dict]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_runtime_device(self.settings.device_override)

    @property
    def loaded_model_count(self) -> int:
        return sum(len(task_models) for task_models in self.models.values())

    def load(self) -> None:
        manifests = load_required_manifests(
            self.settings.checkpoint_root,
            strict=self.settings.strict_model_loading,
        )
        loaded: dict[str, dict[str, dict]] = {}

        for task, task_manifests in manifests.items():
            loaded[task] = {}
            for model_name, manifest in task_manifests.items():
                family = manifest["family"]
                if family == "transformer":
                    bundle = load_transformer_bundle(
                        manifest,
                        self.settings.checkpoint_root,
                        self.device,
                    )
                elif family == "siamese":
                    bundle = load_siamese_bundle(
                        manifest,
                        self.settings.checkpoint_root,
                        self.device,
                    )
                elif family == "sbert":
                    bundle = load_sbert_bundle(manifest, self.device)
                else:
                    raise ValueError(f"Unsupported model family: {family}")
                loaded[task][model_name] = bundle

        self.models = loaded
        self.ready = self.loaded_model_count > 0

    def public_inventory(self) -> dict[str, list[str]]:
        return {
            "paraphrase_detection": list(self.models.get("paraphrase_detection", {}).keys()),
            "semantic_similarity": list(self.models.get("semantic_similarity", {}).keys()),
        }

    def predict(self, sentence1: str, sentence2: str) -> dict:
        sentence1 = normalise_sentence(sentence1.strip())
        sentence2 = normalise_sentence(sentence2.strip())

        paraphrase = {
            model_name: self._predict_paraphrase(bundle, sentence1, sentence2)
            for model_name, bundle in self.models["paraphrase_detection"].items()
        }
        similarity = {
            model_name: self._predict_similarity(bundle, sentence1, sentence2)
            for model_name, bundle in self.models["semantic_similarity"].items()
        }
        return {
            "input": {
                "sentence1": sentence1,
                "sentence2": sentence2,
            },
            "paraphrase_detection": paraphrase,
            "semantic_similarity": similarity,
        }

    def _predict_paraphrase(self, bundle: dict, sentence1: str, sentence2: str) -> dict:
        manifest = bundle["manifest"]
        family = manifest["family"]

        if family == "transformer":
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            encoding = tokenizer(
                sentence1,
                sentence2,
                padding="max_length",
                truncation=True,
                max_length=manifest["max_len"],
                return_tensors="pt",
            )
            encoding = {key: value.to(self.device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = model(**encoding).logits
        else:
            model = bundle["model"]
            s1 = torch.tensor(
                [self._to_indices(sentence1, bundle["word2idx"], bundle["max_len"])],
                dtype=torch.long,
                device=self.device,
            )
            s2 = torch.tensor(
                [self._to_indices(sentence2, bundle["word2idx"], bundle["max_len"])],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                logits = model(s1, s2)["logits"]

        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        label_map = manifest["label_map"]
        label = label_map[str(int(predicted.item()))]
        return {
            "label": label,
            "confidence": float(confidence.item()),
        }

    def _predict_similarity(self, bundle: dict, sentence1: str, sentence2: str) -> dict:
        manifest = bundle["manifest"]
        family = manifest["family"]

        if family == "transformer":
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            encoding = tokenizer(
                sentence1,
                sentence2,
                padding="max_length",
                truncation=True,
                max_length=manifest["max_len"],
                return_tensors="pt",
            )
            encoding = {key: value.to(self.device) for key, value in encoding.items()}
            with torch.no_grad():
                score = model(**encoding).logits.squeeze(1).item()
        elif family == "siamese":
            model = bundle["model"]
            s1 = torch.tensor(
                [self._to_indices(sentence1, bundle["word2idx"], bundle["max_len"])],
                dtype=torch.long,
                device=self.device,
            )
            s2 = torch.tensor(
                [self._to_indices(sentence2, bundle["word2idx"], bundle["max_len"])],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                score = model(s1, s2)["scores"].item()
        else:
            embeddings = bundle["model"].encode(
                [sentence1, sentence2],
                convert_to_tensor=True,
            )
            cosine_sim = F.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0),
            )
            score = (cosine_sim.item() + 1.0) / 2.0

        return {
            "score": float(max(0.0, min(1.0, score))),
            "scale": manifest["scale"],
        }

    @staticmethod
    def _to_indices(sentence: str, word2idx: dict[str, int], max_len: int) -> list[int]:
        tokens = sentence.split()
        indices = [word2idx.get(token, word2idx.get("<UNK>", 1)) for token in tokens]
        indices = indices[:max_len]
        if len(indices) < max_len:
            indices.extend([word2idx.get("<PAD>", 0)] * (max_len - len(indices)))
        return indices
