from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer

try:
    from src.compare import benchmark_model, benchmark_siamese
    from src.dataset import SentencePairDataset
    from src.glove_utils import build_vocab, load_glove_embeddings
    from src.models import get_bert, get_distilbert, get_roberta
    from src.siamese_dataset import SiameseTextDataset
    from src.siamese_model import SiameseGRU, SiameseLSTM
    from src.train import train_model, train_siamese
    from src.visualize import plot_complexity, plot_metrics
except ModuleNotFoundError:  # pragma: no cover - repo-root import path
    from backend.src.compare import benchmark_model, benchmark_siamese
    from backend.src.dataset import SentencePairDataset
    from backend.src.glove_utils import build_vocab, load_glove_embeddings
    from backend.src.models import get_bert, get_distilbert, get_roberta
    from backend.src.siamese_dataset import SiameseTextDataset
    from backend.src.siamese_model import SiameseGRU, SiameseLSTM
    from backend.src.train import train_model, train_siamese
    from backend.src.visualize import plot_complexity, plot_metrics

from .artifacts import export_siamese_checkpoint, export_transformer_checkpoint
from .common import (
    CHECKPOINT_ROOT,
    PARAPHRASE_LABEL_MAP,
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_dir,
    resolve_training_device,
    write_json,
)


SIAMESE_HIDDEN_DIM = 128
SIAMESE_MAX_LEN = 64
TRANSFORMER_MAX_LEN = 128


def train_and_export_mrpc(checkpoint_root: Path | None = None) -> dict:
    checkpoint_root = checkpoint_root or CHECKPOINT_ROOT
    manifest_root = ensure_dir(checkpoint_root / "manifests")
    output_root = ensure_dir(checkpoint_root / "paraphrase" / "mrpc")
    device = resolve_training_device()

    print("Using device:", device)
    print("\nLoading MRPC dataset...")
    dataset = load_dataset("glue", "mrpc")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    results = {}

    all_sentences = list(train_data["sentence1"]) + list(train_data["sentence2"])
    word2idx, _ = build_vocab(all_sentences)
    embedding_matrix = load_glove_embeddings(word2idx)

    siamese_train = SiameseTextDataset(
        train_data["sentence1"],
        train_data["sentence2"],
        train_data["label"],
        word2idx,
        max_len=SIAMESE_MAX_LEN,
        task="classification",
    )
    siamese_val = SiameseTextDataset(
        val_data["sentence1"],
        val_data["sentence2"],
        val_data["label"],
        word2idx,
        max_len=SIAMESE_MAX_LEN,
        task="classification",
    )
    siamese_train_loader = DataLoader(siamese_train, batch_size=32, shuffle=True)
    siamese_val_loader = DataLoader(siamese_val, batch_size=32)

    siamese_models = {
        "Siamese-LSTM": SiameseLSTM(
            embedding_matrix,
            hidden_dim=SIAMESE_HIDDEN_DIM,
            task="classification",
        ),
        "Siamese-GRU": SiameseGRU(
            embedding_matrix,
            hidden_dim=SIAMESE_HIDDEN_DIM,
            task="classification",
        ),
    }

    for name, model in siamese_models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        for epoch in range(5):
            loss = train_siamese(model, siamese_train_loader, optimizer, device)
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}")

        result = benchmark_siamese(model, siamese_val_loader, device)
        results[name] = result
        export_siamese_checkpoint(
            task="paraphrase_detection",
            dataset="mrpc",
            model_name=name,
            model=model.cpu(),
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            embedding_matrix=embedding_matrix,
            word2idx=word2idx,
            max_len=SIAMESE_MAX_LEN,
            hidden_dim=SIAMESE_HIDDEN_DIM,
            num_layers=1,
            dropout=0.3,
            label_map=PARAPHRASE_LABEL_MAP,
            scale="0-1",
            checkpoint_root=checkpoint_root,
            manifest_root=manifest_root,
        )
        model.to(device)

    transformer_models = {
        "BERT": (get_bert, BertTokenizer, "bert-base-uncased"),
        "RoBERTa": (get_roberta, RobertaTokenizer, "roberta-base"),
        "DistilBERT": (get_distilbert, DistilBertTokenizer, "distilbert-base-uncased"),
    }

    for name, (model_fn, tokenizer_cls, tokenizer_name) in transformer_models.items():
        print(f"\nTraining {name}...")
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        train_dataset = SentencePairDataset(
            train_data["sentence1"],
            train_data["sentence2"],
            train_data["label"],
            tokenizer,
            max_len=TRANSFORMER_MAX_LEN,
        )
        val_dataset = SentencePairDataset(
            val_data["sentence1"],
            val_data["sentence2"],
            val_data["label"],
            tokenizer,
            max_len=TRANSFORMER_MAX_LEN,
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        model = model_fn().to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        for epoch in range(2):
            loss = train_model(model, train_loader, optimizer, device)
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}")

        result = benchmark_model(model, val_loader, device)
        results[name] = result
        export_transformer_checkpoint(
            task="paraphrase_detection",
            dataset="mrpc",
            model_name=name,
            model=model.cpu(),
            tokenizer=tokenizer,
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            max_len=TRANSFORMER_MAX_LEN,
            label_map=PARAPHRASE_LABEL_MAP,
            scale="0-1",
            checkpoint_root=checkpoint_root,
            manifest_root=manifest_root,
        )
        model.to(device)

    write_json(RESULTS_DIR / "mrpc_results.json", results)
    plot_metrics(results, save_dir=str(PLOTS_DIR / "mrpc"))
    plot_complexity(results, save_dir=str(PLOTS_DIR / "mrpc"))
    return results


def main() -> None:
    train_and_export_mrpc()


if __name__ == "__main__":
    main()
