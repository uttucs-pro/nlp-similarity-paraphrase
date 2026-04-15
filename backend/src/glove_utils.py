"""
GloVe Embedding Utilities

Provides functions to build vocabularies from training sentences
and load pre-trained GloVe embeddings. Used by Siamese LSTM/GRU
models which rely on static (non-contextualised) word representations.
"""

import numpy as np
import os
from collections import Counter
from pathlib import Path

try:
    from src.preprocessing import clean_text, tokenize_sentence
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.preprocessing import clean_text, tokenize_sentence

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
DEFAULT_GLOVE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "glove" / "glove.6B.100d.txt"
)


def build_vocab(sentences, min_freq=1):
    """
    Build a word-to-index vocabulary from a list of sentences.

    Args:
        sentences: list of raw sentence strings
        min_freq: minimum word frequency to include in vocab

    Returns:
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
    """
    counter = Counter()
    for sent in sentences:
        tokens = tokenize_sentence(sent)
        counter.update(tokens)

    # Start with special tokens
    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

    return word2idx, idx2word


def load_glove_embeddings(word2idx, glove_path="data/glove/glove.6B.100d.txt", embed_dim=100):
    """
    Load GloVe vectors and create an embedding matrix aligned with the vocabulary.

    Words not found in GloVe are initialised with small random vectors.
    The PAD token embedding is kept as zeros.

    Args:
        word2idx: dict mapping word -> index
        glove_path: path to GloVe text file
        embed_dim: dimensionality of GloVe vectors (must match file)

    Returns:
        embedding_matrix: numpy array of shape (vocab_size, embed_dim)
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    resolved_glove_path = Path(glove_path)

    if not resolved_glove_path.is_absolute():
        if glove_path == "data/glove/glove.6B.100d.txt":
            resolved_glove_path = DEFAULT_GLOVE_PATH
        else:
            resolved_glove_path = Path.cwd() / resolved_glove_path

    # PAD token gets zero vector
    embedding_matrix[PAD_IDX] = np.zeros(embed_dim)

    if not resolved_glove_path.exists():
        print(f"\n[WARNING] GloVe file not found at: {resolved_glove_path}")
        print("Please download GloVe embeddings:")
        print("  1. Visit: https://nlp.stanford.edu/projects/glove/")
        print("  2. Download glove.6B.zip")
        print(f"  3. Extract glove.6B.100d.txt to {DEFAULT_GLOVE_PATH.parent}/")
        print("Continuing with random embeddings...\n")
        return embedding_matrix

    found = 0
    with open(resolved_glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                idx = word2idx[word]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embed_dim:
                    embedding_matrix[idx] = vector
                    found += 1

    print(f"GloVe: loaded {found}/{vocab_size} word vectors ({found/vocab_size*100:.1f}% coverage)")
    return embedding_matrix


def sentences_to_indices(sentence, word2idx, max_len=128):
    """
    Convert a sentence string to a list of word indices.

    Args:
        sentence: raw sentence string
        word2idx: vocabulary mapping
        max_len: maximum sequence length (pad/truncate)

    Returns:
        indices: list of integer indices of length max_len
    """
    tokens = tokenize_sentence(sentence)
    indices = [word2idx.get(token, UNK_IDX) for token in tokens]

    # Truncate
    if len(indices) > max_len:
        indices = indices[:max_len]

    # Pad
    while len(indices) < max_len:
        indices.append(PAD_IDX)

    return indices
