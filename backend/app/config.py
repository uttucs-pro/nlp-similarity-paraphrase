from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    frontend_origin: str
    checkpoint_root: Path
    device_override: str | None
    strict_model_loading: bool
    max_sentence_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        backend_root = Path(__file__).resolve().parents[1]
        return cls(
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
            checkpoint_root=Path(
                os.getenv("CHECKPOINT_ROOT", str(backend_root / "checkpoints"))
            ).resolve(),
            device_override=os.getenv("DEVICE_OVERRIDE"),
            strict_model_loading=_parse_bool(
                os.getenv("STRICT_MODEL_LOADING"),
                True,
            ),
            max_sentence_chars=int(os.getenv("MAX_SENTENCE_CHARS", "1000")),
        )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.frontend_origin.split(",")
            if origin.strip()
        ]
