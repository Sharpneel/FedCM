from itertools import combinations

import numpy as np


EPS = np.finfo(float).eps


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def _rms_distance(x):
    return np.sqrt(np.mean((x[:, None] - x[None, :]) ** 2, axis=-1))


def distance_views(signatures, *, use_feedback):
    hist = np.stack([s.histogram for s in signatures]).astype(float)
    hist /= hist.sum(axis=1, keepdims=True)
    p, q = hist[:, None], hist[None, :]
    log_mid = np.log((p + q) / 2 + EPS)
    js = (p * (np.log(p + EPS) - log_mid) + q * (np.log(q + EPS) - log_mid)).sum(-1) / 2
    means = _unit(np.stack([s.mean for s in signatures]))
    views = dict(label=js, feature_mean=1 - means @ means.T,
                 feature_variance=_rms_distance(np.log(np.stack([s.variance for s in signatures]) + EPS)))

    prototypes = [s.prototypes for s in signatures]
    classes = set().union(*prototypes)
    if classes:
        centers = {c: np.mean([p[c] for p in prototypes if c in p], axis=0) for c in classes}
        centered = [{c: x - centers[c] for c, x in p.items()} for p in prototypes]
        shifts = np.stack([np.mean(list(p.values()), axis=0) if p else np.zeros_like(means[0])
                           for p in centered])
        views['feature_shift'] = _rms_distance(shifts)
        distance = np.full_like(js, np.nan)
        observed = []
        for i, j in combinations(range(len(centered)), 2):
            a, b = centered[i], centered[j]
            if common := a.keys() & b.keys():
                value = np.mean([1 - _unit(a[c]) @ _unit(b[c]) for c in common])
                distance[i, j] = distance[j, i] = value
                observed.append(value)
        distance[np.isnan(distance)] = np.median(observed) if observed else 0
        views['class_prototype'] = distance

    if use_feedback:
        routes = _unit(np.stack([s.routes for s in signatures]))
        views['route'] = 1 - routes @ routes.T
        views['uncertainty'] = _rms_distance(np.array([[s.uncertainty] for s in signatures]))
    return views


def _scale(distance, quantiles):
    lo, hi = np.quantile(distance, quantiles)
    if hi - lo <= EPS:
        lo, hi = distance.min(), distance.max()
    return np.clip((distance - lo) / max(hi - lo, EPS), 0, 1)


def _normalize(distance, quantiles):
    distance = _scale(distance, quantiles)
    np.fill_diagonal(distance, 0)
    return distance


def _softmax(x):
    weights = np.exp(x - x.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)
