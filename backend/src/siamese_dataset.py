"""
Siamese Text Dataset

PyTorch Dataset for Siamese models that converts raw sentences into
word-index sequences using a pre-built vocabulary. Applies text
preprocessing (cleaning + tokenisation) before conversion.

Supports both classification (integer labels) and regression (float labels).
"""

import torch
from torch.utils.data import Dataset

try:
    from src.preprocessing import clean_text, tokenize_sentence
    from src.glove_utils import UNK_IDX, PAD_IDX
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.preprocessing import clean_text, tokenize_sentence
    from backend.src.glove_utils import UNK_IDX, PAD_IDX


class SiameseTextDataset(Dataset):
    """
    Dataset for Siamese LSTM/GRU models.

    Unlike the transformer SentencePairDataset which uses subword tokenizers,
    this dataset tokenizes sentences into word-level tokens and converts them
    to indices using a pre-built vocabulary (aligned with GloVe embeddings).
    """

    def __init__(self, sentence1, sentence2, labels, word2idx,
                 max_len=64, task="classification"):
        """
        Args:
            sentence1: list of first sentences
            sentence2: list of second sentences
            labels: list of labels (int for classification, float for regression)
            word2idx: vocabulary mapping (word -> index)
            max_len: maximum sequence length (pad/truncate)
            task: 'classification' or 'regression'
        """
        self.s1 = sentence1
        self.s2 = sentence2
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.labels)

    def _tokenize_and_index(self, sentence):
        """Clean, tokenize, and convert a sentence to padded word indices."""
        tokens = tokenize_sentence(sentence)
        indices = [self.word2idx.get(token, UNK_IDX) for token in tokens]

        # Truncate
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]

        # Pad
        while len(indices) < self.max_len:
            indices.append(PAD_IDX)

        return indices

    def __getitem__(self, idx):
        s1_indices = self._tokenize_and_index(self.s1[idx])
        s2_indices = self._tokenize_and_index(self.s2[idx])

        item = {
            "s1_input_ids": torch.tensor(s1_indices, dtype=torch.long),
            "s2_input_ids": torch.tensor(s2_indices, dtype=torch.long),
        }

        if self.task == "classification":
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
