"""
Text Preprocessing Utilities

Provides text cleaning, normalisation, and tokenisation functions.
Used by the Siamese models for word-level tokenisation with static
embeddings, and available for any custom preprocessing pipelines.

Covers the proposal's requirements for:
  - Tokenisation
  - Sentence normalisation
  - Handling out-of-vocabulary terms (via downstream GloVe UNK mapping)
"""

import re


def clean_text(text):
    """
    Basic text cleaning: lowercase + remove non-alphanumeric characters.

    Args:
        text: raw input string

    Returns:
        cleaned string
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def normalise_sentence(text):
    """
    Full sentence normalisation for NLP preprocessing.

    Steps:
      1. Lowercase
      2. Expand common contractions
      3. Remove special characters (keep alphanumeric + spaces)
      4. Collapse multiple whitespace
      5. Strip leading/trailing whitespace

    Args:
        text: raw input string

    Returns:
        normalised string
    """
    text = text.lower()

    # Expand common contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_sentence(text):
    """
    Tokenize a sentence into word-level tokens after normalisation.

    This is used for Siamese models with static embeddings (GloVe),
    where subword tokenisation is not applicable.

    Args:
        text: raw input string

    Returns:
        list of word tokens
    """
    normalised = normalise_sentence(text)
    tokens = normalised.split()
    return tokens
