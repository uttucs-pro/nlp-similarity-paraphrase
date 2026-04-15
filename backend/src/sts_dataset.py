"""
STS-B (Semantic Textual Similarity Benchmark) Dataset

PyTorch Dataset for the STS-B regression task. Unlike the binary
paraphrase detection datasets (MRPC, QQP), STS-B uses continuous
similarity scores from 0.0 to 5.0, making it a regression problem.

Scores are normalised to [0, 1] for training stability.
"""

import torch
from torch.utils.data import Dataset

try:
    from src.preprocessing import normalise_sentence
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.preprocessing import normalise_sentence


class STSDataset(Dataset):
    """
    Dataset for STS-B with transformer tokenizers.

    Produces tokenised sentence pairs with float labels normalised
    to [0, 1] range (original STS-B scores are 0.0 to 5.0).
    """

    def __init__(self, sentence1, sentence2, labels, tokenizer, max_len=128, normalise=True):
        """
        Args:
            sentence1: list of first sentences
            sentence2: list of second sentences
            labels: list of float labels (STS scores, 0.0 - 5.0)
            tokenizer: HuggingFace tokenizer
            max_len: maximum sequence length
        """
        self.s1 = sentence1
        self.s2 = sentence2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.normalise = normalise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence1 = normalise_sentence(self.s1[idx]) if self.normalise else self.s1[idx]
        sentence2 = normalise_sentence(self.s2[idx]) if self.normalise else self.s2[idx]

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # Normalise STS-B score from [0, 5] to [0, 1]
        item['labels'] = torch.tensor(self.labels[idx] / 5.0, dtype=torch.float)

        return item
