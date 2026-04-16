import torch
from tqdm import tqdm


def train_model(model, dataloader, optimizer, device, scheduler=None, accumulation_steps=4):
    """Training loop for HuggingFace transformer models (classification & regression).

    Supports gradient accumulation to simulate larger batch sizes without
    increasing memory usage. Effective batch size = actual_batch * accumulation_steps.
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def train_siamese(model, dataloader, optimizer, device, scheduler=None):
    """
    Training loop for Siamese LSTM/GRU models.

    Unlike transformer models which accept a flat dict, Siamese models
    take separate sentence inputs (s1_input_ids, s2_input_ids).
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        s1 = batch['s1_input_ids'].to(device)
        s2 = batch['s2_input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(s1, s2, labels=labels)

        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

