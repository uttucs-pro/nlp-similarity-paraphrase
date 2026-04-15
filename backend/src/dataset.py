import torch
from torch.utils.data import Dataset

try:
    from src.preprocessing import normalise_sentence
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.preprocessing import normalise_sentence

class SentencePairDataset(Dataset):
    def __init__(self, sentence1, sentence2, labels, tokenizer, max_len=128, normalise=True):
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
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
