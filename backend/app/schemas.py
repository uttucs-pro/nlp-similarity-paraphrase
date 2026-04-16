from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class PredictRequest(BaseModel):
    sentence1: str
    sentence2: str


class InputPayload(BaseModel):
    sentence1: str
    sentence2: str


class ClassificationPrediction(BaseModel):
    label: str
    confidence: float


class SimilarityPrediction(BaseModel):
    score: float
    scale: str


class DatasetResults(BaseModel):
    """Results for a single dataset, with base and tuned variant predictions."""
    task: str  # "paraphrase_detection" or "semantic_similarity"
    base: Dict[str, Any]
    tuned: Dict[str, Any]


class PredictResponse(BaseModel):
    input: InputPayload
    mrpc: DatasetResults
    qqp: DatasetResults
    stsb: DatasetResults


class HealthResponse(BaseModel):
    status: str
    ready: bool
    loaded_models: int


class ModelsResponse(BaseModel):
    mrpc: Dict[str, list[str]]
    qqp: Dict[str, list[str]]
    stsb: Dict[str, list[str]]
