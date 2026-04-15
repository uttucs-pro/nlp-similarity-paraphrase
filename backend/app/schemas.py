from __future__ import annotations

from typing import Dict

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


class PredictResponse(BaseModel):
    input: InputPayload
    paraphrase_detection: Dict[str, ClassificationPrediction]
    semantic_similarity: Dict[str, SimilarityPrediction]


class HealthResponse(BaseModel):
    status: str
    ready: bool
    loaded_models: int


class ModelsResponse(BaseModel):
    paraphrase_detection: list[str]
    semantic_similarity: list[str]
