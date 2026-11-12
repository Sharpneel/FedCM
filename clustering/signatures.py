from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Signature:
    histogram: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    prototypes: dict
    uncertainty: float
    routes: np.ndarray


def summarize(features, labels, histogram, uncertainty, routes, *, min_class_samples, prototype_count):
    """Summarize comparable local features, including those from a reference encoder."""
    classes, counts = np.unique(labels, return_counts=True)
    by_class = {c: features[labels == c] for c in classes}
    balanced = [by_class[c] for c, n in zip(classes, counts) if n >= min_class_samples]
    groups = balanced or [features]
    largest = np.argsort(-counts)[:prototype_count]
    prototypes = {c: by_class[c].mean(axis=0) for c in classes[largest]}
    return Signature(histogram, np.mean([x.mean(axis=0) for x in groups], axis=0),
                     np.mean([x.var(axis=0) for x in groups], axis=0), prototypes,
                     uncertainty, routes)


def client_signatures(loaders, reference_features, num_classes, experts):
    signatures = []
    for loader in loaders:
        features, labels = [], []
        for x, y in loader:
            features.append(reference_features(x))
            labels.append(y)
        x, y = torch.cat(features), torch.cat(labels)
        signatures.append(summarize(
            x.numpy(), y.numpy(), y.bincount(minlength=num_classes).numpy(),
            uncertainty=0.0, routes=np.zeros(experts),
            min_class_samples=3, prototype_count=2,
        ))
    return signatures
