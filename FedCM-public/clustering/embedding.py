import torch
from torch.nn.functional import normalize


def fit_embeddings(projector, signatures, labels, distance, optimizer, steps, *,
                   neighbors, hard_negatives, refresh_every, temperature):
    """Fit client embeddings with same-cluster/neighbor positives and hard negatives."""
    distance = torch.as_tensor(distance, device=signatures.device)
    labels = torch.as_tensor(labels, device=signatures.device)
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    nearest = distance.masked_fill(diagonal, torch.inf).argsort(dim=1)[:, :min(neighbors, len(labels) - 1)]
    positive = (same & ~diagonal).scatter(1, nearest, True)
    projector.train()
    for step in range(steps):
        embedding = normalize(projector(signatures), dim=1)
        similarity = embedding @ embedding.T
        if step % refresh_every == 0:
            ranking = -distance if step == 0 else similarity.detach()
            indices = ranking.masked_fill(same, -torch.inf).argsort(descending=True)[:, :hard_negatives]
            negative = torch.zeros_like(same).scatter(1, indices, True) & ~same
        rows = positive.any(dim=1) & negative.any(dim=1)
        if not rows.any():
            break
        logits = similarity[rows] / temperature
        log_pos = logits.masked_fill(~positive[rows], -torch.inf).logsumexp(dim=1)
        log_neg = logits.masked_fill(~negative[rows], -torch.inf).logsumexp(dim=1)
        loss = (torch.logaddexp(log_pos, log_neg) - log_pos).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    projector.eval()
    with torch.no_grad():
        return normalize(projector(signatures), dim=1).cpu().numpy()
