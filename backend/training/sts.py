from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
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
        base_predict_similarity,
    )
    from src.glove_utils import build_vocab, load_glove_embeddings
    from src.models import get_bert_regression, get_distilbert_regression, get_roberta_regression
    from src.sbert_model import SBERTModel
    from src.siamese_dataset import SiameseTextDataset
    from src.siamese_model import SiameseGRU, SiameseLSTM
    from src.sts_dataset import STSDataset
    from src.sts_evaluate import evaluate_sbert_sts, evaluate_sts, evaluate_sts_siamese
    from src.train import train_model, train_siamese
    from src.visualize import plot_sts_metrics
except ModuleNotFoundError:  # pragma: no cover
    from backend.src.base_inference import (
        BASE_TRANSFORMER_IDS,
        glove_cosine_similarity,
        transformer_cosine_similarity,
        base_predict_similarity,
    )
    from backend.src.glove_utils import build_vocab, load_glove_embeddings
    from backend.src.models import get_bert_regression, get_distilbert_regression, get_roberta_regression
    from backend.src.sbert_model import SBERTModel
    from backend.src.siamese_dataset import SiameseTextDataset
    from backend.src.siamese_model import SiameseGRU, SiameseLSTM
    from backend.src.sts_dataset import STSDataset
    from backend.src.sts_evaluate import evaluate_sbert_sts, evaluate_sts, evaluate_sts_siamese
    from backend.src.train import train_model, train_siamese
    from backend.src.visualize import plot_sts_metrics

from .artifacts import export_base_manifest, export_sbert_manifest, export_siamese_checkpoint, export_transformer_checkpoint
from .common import (
    Variant,
    checkpoint_root_for,
    ensure_dir,
    plots_dir_for,
    resolve_training_device,
    results_dir_for,
    write_json,
)

SIAMESE_HIDDEN_DIM = 256
SIAMESE_MAX_LEN = 80
SIAMESE_EPOCHS = 30
SIAMESE_LR = 2e-4
SIAMESE_PATIENCE = 7
TRANSFORMER_MAX_LEN = 128
TRANSFORMER_EPOCHS = 6
SBERT_MODEL_ID = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Base variant: evaluate pre-trained models on STS-B (no fine-tuning)
# ---------------------------------------------------------------------------

def _evaluate_base_stsb(checkpoint_root: Path) -> dict:
    manifest_root = ensure_dir(checkpoint_root / "base" / "manifests")
    device = resolve_training_device()

    print("\n=== STS-B BASE (Pre-trained, No Fine-tuning) ===")
    print("Using device:", device)

    dataset = load_dataset("glue", "stsb")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    val_s1 = list(val_data["sentence1"])
    val_s2 = list(val_data["sentence2"])
    val_labels = [label / 5.0 for label in val_data["label"]]  # normalise to [0, 1]

    # GloVe for Siamese base
    all_sentences = list(train_data["sentence1"]) + list(train_data["sentence2"])
    word2idx, _ = build_vocab(all_sentences)
    embedding_matrix = load_glove_embeddings(word2idx)

    from scipy.stats import pearsonr, spearmanr

    results = {}

    # Siamese base: GloVe mean-pooling
    for name in ["Siamese-LSTM", "Siamese-GRU"]:
        print(f"\nEvaluating base {name} (GloVe mean-pooling)...")
        preds = [glove_cosine_similarity(s1, s2, word2idx, embedding_matrix) for s1, s2 in zip(val_s1, val_s2)]
        pearson_corr, _ = pearsonr(val_labels, preds)
        spearman_corr, _ = spearmanr(val_labels, preds)
        results[name] = {"pearson": float(pearson_corr), "spearman": float(spearman_corr)}
        print(f"  {name} base: pearson={pearson_corr:.4f}, spearman={spearman_corr:.4f}")

        export_base_manifest(
            dataset="stsb", model_name=name, family="siamese",
            manifest_root=manifest_root,
        )

    # Transformer base: pre-trained encoder + cosine similarity
    from transformers import AutoModel, AutoTokenizer
    for name, model_id in BASE_TRANSFORMER_IDS.items():
        print(f"\nEvaluating base {name} ({model_id}, mean-pooling)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

        preds = [transformer_cosine_similarity(s1, s2, model, tokenizer, device) for s1, s2 in zip(val_s1, val_s2)]
        pearson_corr, _ = pearsonr(val_labels, preds)
        spearman_corr, _ = spearmanr(val_labels, preds)
        results[name] = {"pearson": float(pearson_corr), "spearman": float(spearman_corr)}
        print(f"  {name} base: pearson={pearson_corr:.4f}, spearman={spearman_corr:.4f}")

        export_base_manifest(
            dataset="stsb", model_name=name, family="transformer",
            model_id=model_id, manifest_root=manifest_root,
        )
        del model

    # Note: SBERT excluded from base per user requirement

    res_dir = results_dir_for("base")
    plt_dir = plots_dir_for("base")
    write_json(ensure_dir(res_dir) / "sts_results.json", results)
    plot_sts_metrics(results, save_dir=str(ensure_dir(plt_dir / "sts")))
    return results


# ---------------------------------------------------------------------------
# Tuned variant: fine-tune models on STS-B
# ---------------------------------------------------------------------------

def _train_tuned_stsb(checkpoint_root: Path) -> dict:
    ckpt_root = checkpoint_root_for("tuned")
    manifest_root = ensure_dir(ckpt_root / "manifests")
    output_root = ensure_dir(ckpt_root / "stsb")
    device = resolve_training_device()

    print("\n=== STS-B TUNED (Fine-tuned) ===")
    print("Using device:", device)
    print("\nLoading STS-B dataset...")
    dataset = load_dataset("glue", "stsb")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    results = {}

    all_sentences = list(train_data["sentence1"]) + list(train_data["sentence2"])
    word2idx, _ = build_vocab(all_sentences)
    embedding_matrix = load_glove_embeddings(word2idx)

    siamese_train = SiameseTextDataset(
        train_data["sentence1"], train_data["sentence2"],
        [label / 5.0 for label in train_data["label"]],
        word2idx, max_len=SIAMESE_MAX_LEN, task="regression",
    )
    siamese_val = SiameseTextDataset(
        val_data["sentence1"], val_data["sentence2"],
        [label / 5.0 for label in val_data["label"]],
        word2idx, max_len=SIAMESE_MAX_LEN, task="regression",
    )
    siamese_train_loader = DataLoader(siamese_train, batch_size=32, shuffle=True)
    siamese_val_loader = DataLoader(siamese_val, batch_size=32)

    siamese_models = {
        "Siamese-LSTM": SiameseLSTM(embedding_matrix, hidden_dim=SIAMESE_HIDDEN_DIM, task="regression"),
        "Siamese-GRU": SiameseGRU(embedding_matrix, hidden_dim=SIAMESE_HIDDEN_DIM, task="regression"),
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
            val_result = evaluate_sts_siamese(model, siamese_val_loader, device)
            scheduler.step(val_result["pearson"])
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}  pearson: {val_result['pearson']:.4f}  spearman: {val_result['spearman']:.4f}")

            if val_result["pearson"] > best_metric:
                best_metric = val_result["pearson"]
                best_result = val_result
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
            task="semantic_similarity", dataset="stsb", model_name=name,
            model=model.cpu(),
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            embedding_matrix=embedding_matrix, word2idx=word2idx,
            max_len=SIAMESE_MAX_LEN, hidden_dim=SIAMESE_HIDDEN_DIM,
            num_layers=1, dropout=0.3, label_map=None, scale="0-1",
            checkpoint_root=ckpt_root, manifest_root=manifest_root,
        )
        model.to(device)

    transformer_models = {
        "BERT": (get_bert_regression, BertTokenizer, "bert-base-uncased", 3e-5),
        "RoBERTa": (get_roberta_regression, RobertaTokenizer, "roberta-base", 2e-5),
        "DistilBERT": (get_distilbert_regression, DistilBertTokenizer, "distilbert-base-uncased", 5e-5),
    }

    for name, (model_fn, tokenizer_cls, tokenizer_name, lr) in transformer_models.items():
        print(f"\nTraining {name} ({TRANSFORMER_EPOCHS} epochs, lr={lr})...")
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        train_dataset = STSDataset(
            train_data["sentence1"], train_data["sentence2"], train_data["label"],
            tokenizer, max_len=TRANSFORMER_MAX_LEN,
        )
        val_dataset = STSDataset(
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
            result = evaluate_sts(model, val_loader, device)
            print(f"  Epoch {epoch + 1} loss: {loss:.4f}  pearson: {result['pearson']:.4f}  spearman: {result['spearman']:.4f}")
            if result["pearson"] > best_metric:
                best_metric = result["pearson"]
                best_state = copy.deepcopy(model.state_dict())
                best_result = result

        model.load_state_dict(best_state)
        results[name] = best_result
        export_transformer_checkpoint(
            task="semantic_similarity", dataset="stsb", model_name=name,
            model=model.cpu(), tokenizer=tokenizer,
            output_dir=ensure_dir(output_root / name.lower().replace(" ", "-")),
            max_len=TRANSFORMER_MAX_LEN, label_map=None, scale="0-1",
            checkpoint_root=ckpt_root, manifest_root=manifest_root,
        )
        model.to(device)

    # SBERT (zero-shot, tuned variant only)
    print("\nEvaluating SBERT...")
    sbert = SBERTModel(model_name=SBERT_MODEL_ID)
    sbert_result = evaluate_sbert_sts(
        sbert,
        list(val_data["sentence1"]),
        list(val_data["sentence2"]),
        [label / 5.0 for label in val_data["label"]],
    )
    results["SBERT"] = sbert_result
    export_sbert_manifest(model_name="SBERT", model_id=SBERT_MODEL_ID, manifest_root=manifest_root)

    res_dir = results_dir_for("tuned")
    plt_dir = plots_dir_for("tuned")
    write_json(ensure_dir(res_dir) / "sts_results.json", results)
    plot_sts_metrics(results, save_dir=str(ensure_dir(plt_dir / "sts")))
    return results


def train_and_export_sts(variant: Variant | None = None) -> dict:
    checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
    all_results = {}
    if variant is None or variant == "base":
        all_results["base"] = _evaluate_base_stsb(checkpoint_root)
    if variant is None or variant == "tuned":
        all_results["tuned"] = _train_tuned_stsb(checkpoint_root)
    return all_results


def main() -> None:
    train_and_export_sts()


if __name__ == "__main__":
    main()
