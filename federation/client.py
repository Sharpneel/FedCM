from torch.nn.functional import cross_entropy


def train_local(model, loader, optimizer, epochs, context, temperature):
    """Train on local batches. Context contains the two embeddings and the mask."""
    model.train()
    device = next(model.parameters()).device
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, *_ = model(x.to(device), *context, temperature)
            cross_entropy(logits, y.to(device)).backward()
            optimizer.step()
