from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

try:
    from src.preprocessing import normalise_sentence
    from src.base_inference import (
        glove_cosine_similarity,
        transformer_cosine_similarity,
        base_predict_paraphrase,
        base_predict_similarity,
    )
except ModuleNotFoundError:  # pragma: no cover
    from backend.src.preprocessing import normalise_sentence
    from backend.src.base_inference import (
        glove_cosine_similarity,
        transformer_cosine_similarity,
        base_predict_paraphrase,
        base_predict_similarity,
    )

from .loaders import (
    load_base_glove_bundle,
    load_base_transformer_bundle,
    load_sbert_bundle,
    load_siamese_bundle,
    load_transformer_bundle,
    resolve_runtime_device,
)
from .registry import DATASETS, get_task_for, load_required_manifests


@dataclass
class PredictionService:
    settings: Any
    device: torch.device = field(init=False)
    ready: bool = field(default=False, init=False)
    # models[dataset][variant][model_name] = bundle
    models: dict[str, dict[str, dict[str, dict]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.device = resolve_runtime_device(self.settings.device_override)

    @property
    def loaded_model_count(self) -> int:
        return sum(
            len(variant_models)
            for dataset_models in self.models.values()
            for variant_models in dataset_models.values()
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        manifests = load_required_manifests(
            self.settings.checkpoint_root,
            strict=self.settings.strict_model_loading,
        )
        loaded: dict[str, dict[str, dict[str, dict]]] = {}

        for dataset, variant_manifests in manifests.items():
            loaded[dataset] = {}
            for variant, model_manifests in variant_manifests.items():
                loaded[dataset][variant] = {}
                for model_name, manifest in model_manifests.items():
                    try:
                        bundle = self._load_bundle(manifest, variant)
                        loaded[dataset][variant][model_name] = bundle
                    except Exception as exc:
                        print(f"[WARN] Failed to load {dataset}/{variant}/{model_name}: {exc}")
                        if self.settings.strict_model_loading:
                            raise

        self.models = loaded
        self.ready = self.loaded_model_count > 0

    def _load_bundle(self, manifest: dict, variant: str) -> dict:
        family = manifest["family"]
        inference_mode = manifest.get("inference_mode", "model_forward")

        if variant == "base" or inference_mode == "embedding_similarity":
            # Base models: embedding + cosine similarity
            if family in ("siamese",):
                return load_base_glove_bundle(manifest, self.device)
            else:
                return load_base_transformer_bundle(manifest, self.device)
        else:
            # Tuned models: full model forward pass
            if family == "transformer":
                return load_transformer_bundle(
                    manifest,
                    self.settings.checkpoint_root / "tuned",
                    self.device,
                )
            elif family == "siamese":
                return load_siamese_bundle(
                    manifest,
                    self.settings.checkpoint_root / "tuned",
                    self.device,
                )
            elif family == "sbert":
                return load_sbert_bundle(manifest, self.device)
            else:
                raise ValueError(f"Unsupported model family: {family}")

    # ------------------------------------------------------------------
    # Public inventory
    # ------------------------------------------------------------------

    def public_inventory(self) -> dict:
        inventory = {}
        for dataset in DATASETS:
            inventory[dataset] = {}
            for variant in ("base", "tuned"):
                models = self.models.get(dataset, {}).get(variant, {})
                inventory[dataset][variant] = list(models.keys())
        return inventory

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sentence1: str, sentence2: str) -> dict:
        sentence1 = normalise_sentence(sentence1.strip())
        sentence2 = normalise_sentence(sentence2.strip())

        result = {
            "input": {"sentence1": sentence1, "sentence2": sentence2},
        }

        for dataset in DATASETS:
            task = get_task_for(dataset)
            dataset_result = {"task": task}

            for variant in ("base", "tuned"):
                variant_models = self.models.get(dataset, {}).get(variant, {})
                variant_result = {}

                for model_name, bundle in variant_models.items():
                    manifest = bundle["manifest"]
                    inference_mode = manifest.get("inference_mode", "model_forward")

                    if inference_mode == "embedding_similarity":
                        pred = self._predict_embedding(
                            bundle, sentence1, sentence2, task
                        )
                    elif task == "paraphrase_detection":
                        pred = self._predict_paraphrase(bundle, sentence1, sentence2)
                    else:
                        pred = self._predict_similarity(bundle, sentence1, sentence2)

                    variant_result[model_name] = pred

                dataset_result[variant] = variant_result

            result[dataset] = dataset_result

        return result

    # ------------------------------------------------------------------
    # Base model prediction (embedding cosine similarity)
    # ------------------------------------------------------------------

    def _predict_embedding(
        self, bundle: dict, sentence1: str, sentence2: str, task: str
    ) -> dict:
        manifest = bundle["manifest"]
        family = manifest["family"]

        if family == "siamese":
            sim = glove_cosine_similarity(
                sentence1, sentence2,
                bundle["word2idx"],
                bundle["embedding_matrix"],
            )
        else:
            sim = transformer_cosine_similarity(
                sentence1, sentence2,
                bundle["model"],
                bundle["tokenizer"],
                self.device,
            )

        if task == "paraphrase_detection":
            return base_predict_paraphrase(sim)
        else:
            return base_predict_similarity(sim)

    # ------------------------------------------------------------------
    # Tuned model prediction (model forward pass)
    # ------------------------------------------------------------------

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
