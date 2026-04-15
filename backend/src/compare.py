"""
Model Benchmarking

Provides benchmarking functions for transformer and Siamese models.
Measures accuracy, F1 score, inference time, and model parameter count
for computational complexity comparison (as specified in the proposal).
"""

import time
import torch
from sklearn.metrics import accuracy_score, f1_score


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Returns:
        dict with 'total_params' and 'trainable_params'
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable
    }


def benchmark_model(model, dataloader, device):
    """
    Benchmark a HuggingFace transformer model on classification.

    Returns:
        dict with accuracy, f1, time, and parameter counts
    """
    model.eval()
    preds, labels = [], []

    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    end = time.time()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    result = {
        "accuracy": acc,
        "f1": f1,
        "time": end - start
    }
    result.update(count_parameters(model))

    return result


def benchmark_siamese(model, dataloader, device):
    """
    Benchmark a Siamese LSTM/GRU model on classification.

    Returns:
        dict with accuracy, f1, time, and parameter counts
    """
    model.eval()
    preds, labels = [], []

    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            s1 = batch['s1_input_ids'].to(device)
            s2 = batch['s2_input_ids'].to(device)
            batch_labels = batch['labels']

            outputs = model(s1, s2)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(batch_labels.numpy())

    end = time.time()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    result = {
        "accuracy": acc,
        "f1": f1,
        "time": end - start
    }
    result.update(count_parameters(model))

    return result
