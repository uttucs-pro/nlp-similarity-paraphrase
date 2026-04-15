from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


class SBERTModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def similarity(self, s1, s2):
        emb1 = self.encode(s1)
        emb2 = self.encode(s2)
        return F.cosine_similarity(emb1, emb2)
