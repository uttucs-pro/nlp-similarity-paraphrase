# Backend

FastAPI backend for frontend-backed sentence-pair inference.

## What Was Implemented

- `app/` for the FastAPI service and response schemas
- `inference/` for manifest loading, model loading, and prediction orchestration
- `training/` for MRPC, STS-B, and QQP entrypoints
- strict checkpoint/manifest-based serving
- `POST /predict`, `GET /health`, and `GET /models`

## Important Limitation

The API is implemented, but it will not start serving predictions until the
required checkpoints and manifests exist under `backend/checkpoints/`.

That is intentional: the service uses strict artifact loading and does not fall
back to benchmark result JSON files.

## Required Training/Export Step

You need to run these two commands when you are ready to generate serving
artifacts:

### 1. Export MRPC paraphrase checkpoints

From the repo root:

```bash
MPLCONFIGDIR=/tmp/matplotlib HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 backend/venv/bin/python backend/main.py
```

Or from inside `backend/`:

```bash
MPLCONFIGDIR=/tmp/matplotlib HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python main.py
```

### 2. Export STS-B similarity checkpoints

From the repo root:

```bash
MPLCONFIGDIR=/tmp/matplotlib HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 backend/venv/bin/python backend/run_sts.py
```

Or from inside `backend/`:

```bash
MPLCONFIGDIR=/tmp/matplotlib HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python run_sts.py
```

These commands will populate:

- `checkpoints/paraphrase/mrpc/`
- `checkpoints/sts/`
- `checkpoints/manifests/`

## Start The API

After the checkpoint export step succeeds:

From the repo root:

```bash
backend/venv/bin/python backend/run_api.py
```

Or from inside `backend/`:

```bash
venv/bin/python run_api.py
```

The frontend should point `VITE_API_URL` to:

```text
http://127.0.0.1:8000
```

## Public API

### `POST /predict`

Request:

```json
{
  "sentence1": "How can I learn NLP quickly?",
  "sentence2": "What is the fastest way to study NLP?"
}
```

Response shape:

```json
{
  "input": {
    "sentence1": "how can i learn nlp quickly",
    "sentence2": "what is the fastest way to study nlp"
  },
  "paraphrase_detection": {
    "Siamese-LSTM": { "label": "Paraphrase", "confidence": 0.81 },
    "Siamese-GRU": { "label": "Paraphrase", "confidence": 0.76 },
    "BERT": { "label": "Paraphrase", "confidence": 0.91 },
    "RoBERTa": { "label": "Paraphrase", "confidence": 0.94 },
    "DistilBERT": { "label": "Paraphrase", "confidence": 0.89 }
  },
  "semantic_similarity": {
    "Siamese-LSTM": { "score": 0.61, "scale": "0-1" },
    "Siamese-GRU": { "score": 0.72, "scale": "0-1" },
    "BERT": { "score": 0.88, "scale": "0-1" },
    "RoBERTa": { "score": 0.9, "scale": "0-1" },
    "DistilBERT": { "score": 0.86, "scale": "0-1" },
    "SBERT": { "score": 0.84, "scale": "0-1" }
  }
}
```
