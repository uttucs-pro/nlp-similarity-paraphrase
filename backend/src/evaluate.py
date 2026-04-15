import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return acc, f1
