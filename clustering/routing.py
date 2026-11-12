from dataclasses import dataclass

import numpy as np
import torch

from .distances import EPS, _scale, _softmax, _unit


@dataclass
class RoutingContext:
    clients: np.ndarray
    clusters: np.ndarray
    masks: np.ndarray
    membership: np.ndarray
    confidence: np.ndarray

    def for_client(self, index):
        return tuple(torch.as_tensor(x[index]) for x in (self.clients, self.clusters, self.masks))


def routing_context(partition, embeddings, num_experts, *, quantiles, temperature,
                    neighbors, smoothing, neighbor_experts, link_weight):
    n = len(embeddings)
    masks = np.zeros((n, num_experts), dtype=bool)
    membership = np.zeros_like(masks, dtype=float)
    if partition is None:
        zeros = np.zeros_like(embeddings)
        return RoutingContext(zeros, zeros, masks, membership, np.ones(n))

    labels, distance = partition
    groups = [np.flatnonzero(labels == q) for q in np.unique(labels)]
    centers = np.stack([embeddings[group].mean(axis=0) for group in groups])
    to_cluster = np.stack([distance[:, group].mean(axis=1) for group in groups], axis=1)
    for q, group in enumerate(groups):
        if len(group) > 1:
            to_cluster[group, q] *= len(group) / (len(group) - 1)  # Leave out the client itself.
    to_cluster = _scale(to_cluster, quantiles)
    soft = _softmax(-to_cluster / temperature)

    similarity = _unit(embeddings) @ _unit(embeddings).T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argsort(-similarity, axis=1)[:, :min(neighbors, n - 1)]
    if nearest.shape[1]:
        soft = (1 - smoothing) * soft + smoothing * soft[nearest].mean(axis=1)
    membership[:, :len(groups)] = soft
    primary = soft.argmax(axis=1)
    best, second = np.sort(to_cluster, axis=1)[:, :2].T
    confidence = np.where(second > EPS, 1 - best / np.maximum(second, EPS), 0)

    links = np.zeros((len(groups), len(groups)))
    np.add.at(links, (primary[:, None], primary[nearest]), 1)
    sizes = np.bincount(primary, minlength=len(groups))
    similarity = _unit(centers) @ _unit(centers).T + link_weight * links / np.maximum(sizes[:, None], 1)
    np.fill_diagonal(similarity, -np.inf)
    nearby = np.argsort(-similarity, axis=1)[:, :min(neighbor_experts, len(groups) - 1)]
    masks[np.arange(n), primary] = True
    masks[np.arange(n)[:, None], nearby[primary]] = True
    return RoutingContext(embeddings, centers[primary], masks, membership, confidence)
