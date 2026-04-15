from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .schemas import (
    HealthResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)

try:
    from inference.service import PredictionService
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.inference.service import PredictionService


def _validate_sentence(value: str, field_name: str, max_chars: int) -> str:
    text = value.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must not be empty or whitespace only.",
        )
    if len(text) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must not exceed {max_chars} characters.",
        )
    return text


def create_app() -> FastAPI:
    settings = Settings.from_env()
    service = PredictionService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load()
        app.state.settings = settings
        app.state.prediction_service = service
        yield

    app = FastAPI(
        title="NLP Sentence Similarity API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        prediction_service: PredictionService = request.app.state.prediction_service
        return HealthResponse(
            status="ok" if prediction_service.ready else "unavailable",
            ready=prediction_service.ready,
            loaded_models=prediction_service.loaded_model_count,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models(request: Request) -> ModelsResponse:
        prediction_service: PredictionService = request.app.state.prediction_service
        inventory = prediction_service.public_inventory()
        return ModelsResponse(**inventory)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        settings: Settings = request.app.state.settings
        prediction_service: PredictionService = request.app.state.prediction_service

        sentence1 = _validate_sentence(payload.sentence1, "sentence1", settings.max_sentence_chars)
        sentence2 = _validate_sentence(payload.sentence2, "sentence2", settings.max_sentence_chars)

        if not prediction_service.ready:
            raise HTTPException(status_code=503, detail="Prediction service is not ready.")

        result = prediction_service.predict(sentence1, sentence2)
        return PredictResponse(**result)

    return app


app = create_app()
