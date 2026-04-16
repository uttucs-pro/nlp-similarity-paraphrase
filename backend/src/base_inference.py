"""
Base Model Inference (Pre-trained, No Fine-tuning)

Provides embedding-based inference for models that have NOT been
fine-tuned on any task-specific dataset. Instead of using a trained
classification head, these functions:

  1. Encode each sentence independently into a dense vector
  2. Compute cosine similarity between the two vectors
  3. Threshold (for classification) or return directly (for regression)

Supports two embedding families:
  - Static GloVe embeddings (mean-pooled) for Siamese baselines
  - Pre-trained transformer hidden states (mean-pooled) for BERT/RoBERTa/DistilBERT
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from src.preprocessing import normalise_sentence, tokenize_sentence
    from src.glove_utils import UNK_IDX, PAD_IDX
except ModuleNotFoundError:  # pragma: no cover
    from backend.src.preprocessing import normalise_sentence, tokenize_sentence
    from backend.src.glove_utils import UNK_IDX, PAD_IDX


# ---------------------------------------------------------------------------
# GloVe-based embeddings (for Siamese-LSTM / Siamese-GRU base)
# ---------------------------------------------------------------------------

def glove_mean_embedding(sentence: str, word2idx: dict, embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a mean-pooled GloVe embedding for a sentence.

    Args:
        sentence: raw text
        word2idx: vocabulary mapping
        embedding_matrix: (vocab_size, embed_dim)

    Returns:
        (embed_dim,) numpy array
    """
    tokens = tokenize_sentence(sentence)
    indices = [word2idx.get(token, UNK_IDX) for token in tokens]
    # Filter out PAD indices (shouldn't be present, but just in case)
    indices = [idx for idx in indices if idx != PAD_IDX]

    if not indices:
        # If no valid tokens, return zeros
        return np.zeros(embedding_matrix.shape[1], dtype=np.float32)

    vectors = embedding_matrix[indices]  # (num_tokens, embed_dim)
    return np.mean(vectors, axis=0).astype(np.float32)


def glove_cosine_similarity(
    sentence1: str,
    sentence2: str,
    word2idx: dict,
    embedding_matrix: np.ndarray,
) -> float:
    """Compute cosine similarity between two sentences using GloVe mean-pooling."""
    emb1 = glove_mean_embedding(sentence1, word2idx, embedding_matrix)
    emb2 = glove_mean_embedding(sentence2, word2idx, embedding_matrix)

    t1 = torch.tensor(emb1).unsqueeze(0)
    t2 = torch.tensor(emb2).unsqueeze(0)

    sim = F.cosine_similarity(t1, t2).item()
    # Map from [-1, 1] to [0, 1]
    return (sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Transformer-based embeddings (for BERT / RoBERTa / DistilBERT base)
# ---------------------------------------------------------------------------

def transformer_mean_embedding(
    sentence: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_len: int = 128,
) -> torch.Tensor:
    """
    Compute mean-pooled hidden states from a pre-trained transformer.

    Uses the last hidden state, masked to exclude padding tokens,
    then mean-pools over the sequence length dimension.

    Args:
        sentence: normalised text
        model: pre-trained transformer (AutoModel, NOT ForSequenceClassification)
        tokenizer: corresponding tokenizer
        device: torch device
        max_len: max token length

    Returns:
        (hidden_dim,) tensor on CPU
    """
    encoding = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    # last_hidden_state: (1, seq_len, hidden_dim)
    hidden_states = outputs.last_hidden_state

    # Mask out padding tokens
    attention_mask = encoding["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
    masked_hidden = hidden_states * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)  # (1, hidden_dim)
    count = attention_mask.sum(dim=1).clamp(min=1)  # (1, 1)
    mean_pooled = (sum_hidden / count).squeeze(0)  # (hidden_dim,)

    return mean_pooled.cpu()


def transformer_cosine_similarity(
    sentence1: str,
    sentence2: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_len: int = 128,
) -> float:
    """Compute cosine similarity between two sentences using a pre-trained transformer."""
    emb1 = transformer_mean_embedding(sentence1, model, tokenizer, device, max_len)
    emb2 = transformer_mean_embedding(sentence2, model, tokenizer, device, max_len)

    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    # Map from [-1, 1] to [0, 1]
    return (sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Prediction helpers (shared by both families)
# ---------------------------------------------------------------------------

PARAPHRASE_THRESHOLD = 0.5


def base_predict_paraphrase(sim_score: float, threshold: float = PARAPHRASE_THRESHOLD) -> dict:
    """
    Convert a cosine similarity score into a paraphrase classification.

    Args:
        sim_score: similarity in [0, 1]
        threshold: decision boundary

    Returns:
        dict with 'label' and 'confidence'
    """
    if sim_score >= threshold:
        return {"label": "Paraphrase", "confidence": round(sim_score, 4)}
    else:
        return {"label": "Not Paraphrase", "confidence": round(1.0 - sim_score, 4)}


def base_predict_similarity(sim_score: float) -> dict:
    """
    Convert a cosine similarity score into a similarity prediction.

    Args:
        sim_score: similarity in [0, 1]

    Returns:
        dict with 'score' and 'scale'
    """
    return {"score": round(max(0.0, min(1.0, sim_score)), 4), "scale": "0-1"}


# ---------------------------------------------------------------------------
# Pre-trained model loaders (used by inference service for base variant)
# ---------------------------------------------------------------------------

# HuggingFace model IDs for base (pre-trained, no fine-tuning) usage
BASE_TRANSFORMER_IDS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
}


def load_base_transformer(model_id: str, device: torch.device) -> dict:
    """
    Load a pre-trained transformer encoder (AutoModel, not ForSequenceClassification).

    Returns a bundle dict compatible with the inference service.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_id": model_id,
    }
