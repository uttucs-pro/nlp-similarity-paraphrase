from tqdm import tqdm


def train_model(model, dataloader, optimizer, device):
    """Training loop for HuggingFace transformer models (classification & regression)."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_siamese(model, dataloader, optimizer, device):
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
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
