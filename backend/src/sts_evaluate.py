"""
STS-B Evaluation Metrics

Evaluation functions for the Semantic Textual Similarity task using
Pearson and Spearman correlation coefficients, as specified in the
project proposal.

Pearson correlation measures linear correlation between predicted
and true similarity scores. Spearman rank correlation measures
monotonic relationships (rank-order agreement).
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr


def evaluate_sts(model, dataloader, device):
    """
    Evaluate a transformer regression model on STS-B.

    Args:
        model: HuggingFace model with num_labels=1 (regression)
        dataloader: DataLoader with STSDataset
        device: torch device

    Returns:
        dict with 'pearson' and 'spearman' correlation coefficients
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            # For regression models, logits has shape (batch, 1)
            predictions = outputs.logits.squeeze(1)
            predictions = predictions.clamp(0.0, 1.0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pearson_corr, _ = pearsonr(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)

    return {
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr)
    }


def evaluate_sts_siamese(model, dataloader, device):
    """
    Evaluate a Siamese model on STS-B.

    Args:
        model: SiameseLSTM or SiameseGRU in regression mode
        dataloader: DataLoader with SiameseTextDataset (regression)
        device: torch device

    Returns:
        dict with 'pearson' and 'spearman' correlation coefficients
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            s1 = batch['s1_input_ids'].to(device)
            s2 = batch['s2_input_ids'].to(device)
            labels = batch['labels']

            outputs = model(s1, s2)
            predictions = outputs['scores']

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pearson_corr, _ = pearsonr(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)

    return {
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr)
    }


def evaluate_sbert_sts(sbert_model, sentence1_list, sentence2_list, labels):
    """
    Evaluate SBERT on STS-B using zero-shot cosine similarity.

    No fine-tuning required — SBERT directly computes sentence embeddings
    and cosine similarity between pairs.

    Args:
        sbert_model: SBERTModel instance
        sentence1_list: list of first sentences
        sentence2_list: list of second sentences
        labels: list of gold similarity scores (normalised to [0, 1])

    Returns:
        dict with 'pearson' and 'spearman' correlation coefficients
    """
    similarities = sbert_model.similarity(sentence1_list, sentence2_list)
    preds = similarities.cpu().numpy()
    labels = np.array(labels)

    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)

    return {
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr)
    }
