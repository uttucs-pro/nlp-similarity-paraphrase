# Frontend

React + Vite frontend for the NLP sentence-pair demo.

## What It Does

- Accepts two user sentences
- Sends them to the backend with `POST /predict`
- Displays paraphrase detection outputs for:
  - `Siamese-LSTM`
  - `Siamese-GRU`
  - `BERT`
  - `RoBERTa`
  - `DistilBERT`
- Displays semantic similarity outputs for:
  - `Siamese-LSTM`
  - `Siamese-GRU`
  - `BERT`
  - `RoBERTa`
  - `DistilBERT`
  - `SBERT`

## Run Locally

1. Create an env file:
   - copy `.env.example` to `.env`
2. Set `VITE_API_URL` to your backend base URL
3. Start the dev server:

```bash
npm run dev
```

## Expected Backend Response

```json
{
  "input": {
    "sentence1": "string",
    "sentence2": "string"
  },
  "paraphrase_detection": {
    "Siamese-LSTM": { "label": "Paraphrase", "confidence": 0.81 },
    "Siamese-GRU": { "label": "Not Paraphrase", "confidence": 0.64 },
    "BERT": { "label": "Paraphrase", "confidence": 0.92 },
    "RoBERTa": { "label": "Paraphrase", "confidence": 0.95 },
    "DistilBERT": { "label": "Paraphrase", "confidence": 0.9 }
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
