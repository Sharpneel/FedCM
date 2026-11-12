import numpy as np
import torch

from .client import train_local
from .exchange import ClientUpdate, load_shared, shared_state
from .server import aggregate


def run_round(server, clients, loaders, optimizers, context, temperature):
    updates = []
    for i, (client, loader, optimizer) in enumerate(zip(clients, loaders, optimizers)):
        mask = context.masks[i]
        load_shared(client, shared_state(server, mask))
        train_local(client, loader, optimizer, epochs=1, context=context.for_client(i),
                    temperature=temperature)
        updates.append(ClientUpdate(shared_state(client, mask), len(loader.dataset),
                                    context.membership[i], context.confidence[i]))
    active = np.flatnonzero(context.masks.any(axis=0))
    aggregate(server, updates, active, confidence_floor=0.025, weight_global_by_confidence=True)
    return len(active)


@torch.inference_mode()
def test_accuracy(server, clients, loaders, context, temperature):
    correct = total = 0
    for i, (client, loader) in enumerate(zip(clients, loaders)):
        load_shared(client, shared_state(server, context.masks[i]))
        client.eval()
        device = next(client.parameters()).device
        client_context = context.for_client(i)
        for x, y in loader:
            logits, *_ = client(x.to(device), *client_context, temperature)
            correct += (logits.argmax(dim=1) == y.to(device)).sum().item()
            total += len(y)
    return correct / total
