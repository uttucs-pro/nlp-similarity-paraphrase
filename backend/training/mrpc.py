from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from src.base_inference import (
        BASE_TRANSFORMER_IDS,
        glove_cosine_similarity,
        transformer_cosine_similarity,
        base_predict_paraphrase,
    )
    from src.compare import benchmark_model, benchmark_siamese
    from src.dataset import SentencePairDataset
    from src.glove_utils import build_vocab, load_glove_embeddings
    from src.models import get_bert, get_distilbert, get_roberta
    from src.siamese_dataset import SiameseTextDataset
    from src.siamese_model import SiameseGRU, SiameseLSTM
    from src.train import train_model, train_siamese
    from src.visualize import plot_complexity, plot_metrics
except ModuleNotFoundError:  # pragma: no cover
    from backend.src.base_inference import (
        BASE_TRANSFORMER_IDS,
        glove_cosine_similarity,
        transformer_cosine_similarity,
        base_predict_paraphrase,
    )
    from backend.src.compare import benchmark_model, benchmark_siamese
    from backend.src.dataset import SentencePairDataset
    from backend.src.glove_utils import build_vocab, load_glove_embeddings
    from backend.src.models import get_bert, get_distilbert, get_roberta
    from backend.src.siamese_dataset import SiameseTextDataset
    from backend.src.siamese_model import SiameseGRU, SiameseLSTM
    from backend.src.train import train_model, train_siamese
    from backend.src.visualize import plot_complexity, plot_metrics

from .artifacts import export_base_manifest, export_siamese_checkpoint, export_transformer_checkpoint
from .common import (
    PARAPHRASE_LABEL_MAP,
    Variant,
    checkpoint_root_for,
    ensure_dir,
    plots_dir_for,
    resolve_training_device,
    results_dir_for,
    write_json,
)

# Tuned hyperparameters (current optimized config)
SIAMESE_HIDDEN_DIM = 256
SIAMESE_MAX_LEN = 80
SIAMESE_EPOCHS = 30
SIAMESE_LR = 2e-4
SIAMESE_PATIENCE = 7
TRANSFORMER_MAX_LEN = 128
TRANSFORMER_EPOCHS = 6


# ---------------------------------------------------------------------------
# Base variant: evaluate pre-trained models (no fine-tuning)
# ---------------------------------------------------------------------------

def _evaluate_base_mrpc(checkpoint_root: Path) -> dict:
    """Evaluate all base (pre-trained, no fine-tuning) models on MRPC validation."""
    manifest_root = ensure_dir(checkpoint_root / "base" / "manifests")
    device = resolve_training_device()

    print("\n=== MRPC BASE (Pre-trained, No Fine-tuning) ===")
    print("Using device:", device)

    dataset = load_dataset("glue", "mrpc")
    val_data = dataset["validation"]
    val_s1 = list(val_data["sentence1"])
    val_s2 = list(val_data["sentence2"])
    val_labels = list(val_data["label"])

    # Build vocab + GloVe for Siamese base eval
    train_data = dataset["train"]
    all_sentences = list(train_data["sentence1"]) + list(train_data["sentence2"])
    word2idx, _ = build_vocab(all_sentences)
    embedding_matrix = load_glove_embeddings(word2idx)

    results = {}

    # Siamese base: GloVe mean-pooling + cosine similarity
    for name in ["Siamese-LSTM", "Siamese-GRU"]:
        print(f"\nEvaluating base {name} (GloVe mean-pooling)...")
        preds = []
        for s1, s2 in zip(val_s1, val_s2):
            sim = glove_cosine_similarity(s1, s2, word2idx, embedding_matrix)
            pred = base_predict_paraphrase(sim)
            preds.append(1 if pred["label"] == "Paraphrase" else 0)

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds)
        results[name] = {"accuracy": acc, "f1": f1, "time": 0.0, "total_params": 0, "trainable_params": 0}
        print(f"  {name} base: acc={acc:.4f}, f1={f1:.4f}")

        export_base_manifest(
            dataset="mrpc", model_name=name, family="siamese",
            glove_path=str(embedding_matrix),  # reference
            manifest_root=manifest_root,
        )

    # Transformer base: pre-trained encoder + cosine similarity
    from transformers import AutoModel, AutoTokenizer
    for name, model_id in BASE_TRANSFORMER_IDS.items():
        print(f"\nEvaluating base {name} ({model_id}, mean-pooling)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        preds = []
        for s1, s2 in zip(val_s1, val_s2):
            sim = transformer_cosine_similarity(s1, s2, model, tokenizer, device)
            pred = base_predict_paraphrase(sim)
            preds.append(1 if pred["label"] == "Paraphrase" else 0)

        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds)
        total_params = sum(p.numel() for p in model.parameters())
        results[name] = {"accuracy": acc, "f1": f1, "time": 0.0, "total_params": total_params, "trainable_params": 0}
        print(f"  {name} base: acc={acc:.4f}, f1={f1:.4f}")

        export_base_manifest(
            dataset="mrpc", model_name=name, family="transformer",
            model_id=model_id, manifest_root=manifest_root,
        )
        del model

    res_dir = results_dir_for("base")
    plt_dir = plots_dir_for("base")
    write_json(ensure_dir(res_dir) / "mrpc_results.json", results)
    plot_metrics(results, save_dir=str(ensure_dir(plt_dir / "mrpc")))
    return results


# ---------------------------------------------------------------------------
# Tuned variant: fine-tune models (existing logic)
# ---------------------------------------------------------------------------

def _train_tuned_mrpc(checkpoint_root: Path) -> dict:
    """Train and export all tuned (fine-tuned) models on MRPC."""
    ckpt_root = checkpoint_root_for("tuned")
    manifest_root = ensure_dir(ckpt_root / "manifests")
    output_root = ensure_dir(ckpt_root / "mrpc")
    device = resolve_training_device()

    print("\n=== MRPC TUNED (Fine-tuned) ===")
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
        train_data["sentence1"], train_data["sentence2"], train_data["label"],
        word2idx, max_len=SIAMESE_MAX_LEN, task="classification",
    )
    siamese_val = SiameseTextDataset(
        val_data["sentence1"], val_data["sentence2"], val_data["label"],
        word2idx, max_len=SIAMESE_MAX_LEN, task="classification",
    )
    siamese_train_loader = DataLoader(siamese_train, batch_size=32, shuffle=True)
    siamese_val_loader = DataLoader(siamese_val, batch_size=32)

    siamese_models = {
        "Siamese-LSTM": SiameseLSTM(embedding_matrix, hidden_dim=SIAMESE_HIDDEN_DIM, task="classification"),
        "Siamese-GRU": SiameseGRU(embedding_matrix, hidden_dim=SIAMESE_HIDDEN_DIM, task="classification"),
    }

    for name, model in siamese_models.items():
        print(f"\nTraining {name} (max {SIAMESE_EPOCHS} epochs, patience={SIAMESE_PATIENCE})...")
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=SIAMESE_LR, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        best_result = None
        best_metric = -float("inf")
        patience_counter = 0

        for epoch in range(SIAMESE_EPOCHS):
            loss = train_siamese(model, siamese_train_loader, optimizer, device)
            result = benchmark_siamese(model, siamese_val_loader, device)
            scheduler.step(result["accuracy"])
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}  acc: {result['accuracy']:.4f}  f1: {result['f1']:.4f}")

            if result["accuracy"] > best_metric:
                best_metric = result["accuracy"]
                best_result = result
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= SIAMESE_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_state)
        results[name] = best_result
        export_siamese_checkpoint(
            task="paraphrase_detection", dataset="mrpc", model_name=name,
            model=model.cpu(),
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            embedding_matrix=embedding_matrix, word2idx=word2idx,
            max_len=SIAMESE_MAX_LEN, hidden_dim=SIAMESE_HIDDEN_DIM,
            num_layers=1, dropout=0.3, label_map=PARAPHRASE_LABEL_MAP,
            scale="0-1", checkpoint_root=ckpt_root, manifest_root=manifest_root,
        )
        model.to(device)

    transformer_models = {
        "BERT": (get_bert, BertTokenizer, "bert-base-uncased", 3e-5),
        "RoBERTa": (get_roberta, RobertaTokenizer, "roberta-base", 2e-5),
        "DistilBERT": (get_distilbert, DistilBertTokenizer, "distilbert-base-uncased", 5e-5),
    }

    for name, (model_fn, tokenizer_cls, tokenizer_name, lr) in transformer_models.items():
        print(f"\nTraining {name} ({TRANSFORMER_EPOCHS} epochs, lr={lr})...")
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        train_dataset = SentencePairDataset(
            train_data["sentence1"], train_data["sentence2"], train_data["label"],
            tokenizer, max_len=TRANSFORMER_MAX_LEN,
        )
        val_dataset = SentencePairDataset(
            val_data["sentence1"], val_data["sentence2"], val_data["label"],
            tokenizer, max_len=TRANSFORMER_MAX_LEN,
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        model = model_fn().to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        accumulation_steps = 4
        total_steps = (len(train_loader) // accumulation_steps) * TRANSFORMER_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
        )

        best_metric = -float("inf")
        best_state = None
        best_result = None
        for epoch in range(TRANSFORMER_EPOCHS):
            loss = train_model(model, train_loader, optimizer, device, scheduler=scheduler)
            result = benchmark_model(model, val_loader, device)
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}  acc: {result['accuracy']:.4f}  f1: {result['f1']:.4f}")
            if result["accuracy"] > best_metric:
                best_metric = result["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                best_result = result

        model.load_state_dict(best_state)
        results[name] = best_result
        export_transformer_checkpoint(
            task="paraphrase_detection", dataset="mrpc", model_name=name,
            model=model.cpu(), tokenizer=tokenizer,
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            max_len=TRANSFORMER_MAX_LEN, label_map=PARAPHRASE_LABEL_MAP,
            scale="0-1", checkpoint_root=ckpt_root, manifest_root=manifest_root,
        )
        model.to(device)

    res_dir = results_dir_for("tuned")
    plt_dir = plots_dir_for("tuned")
    write_json(ensure_dir(res_dir) / "mrpc_results.json", results)
    plot_metrics(results, save_dir=str(ensure_dir(plt_dir / "mrpc")))
    plot_complexity(results, save_dir=str(ensure_dir(plt_dir / "mrpc")))
    return results


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def train_and_export_mrpc(variant: Variant | None = None) -> dict:
    """Train/evaluate MRPC models. If variant is None, runs both."""
    checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
    all_results = {}

    if variant is None or variant == "base":
        all_results["base"] = _evaluate_base_mrpc(checkpoint_root)

    if variant is None or variant == "tuned":
        all_results["tuned"] = _train_tuned_mrpc(checkpoint_root)

    return all_results


def main() -> None:
    train_and_export_mrpc()


if __name__ == "__main__":
    main()
