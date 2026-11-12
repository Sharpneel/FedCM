import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .distances import _normalize, _softmax


def _capped_softmax(scores, caps):
    weights = np.zeros_like(scores)
    free = np.ones(len(scores), dtype=bool)
    while free.any():
        weights[free] = (1 - weights[~free].sum()) * _softmax(scores[free])
        excess = free & (weights > caps)
        if not excess.any():
            break
        weights[excess] = caps[excess]
        free[excess] = False
    return weights


def select_partition(views, cluster, counts, resamples, rng, *, quantiles, view_temperature,
                     view_caps, imbalance_weight, instability_weight, consensus_weight,
                     cost_weight, permutations, significance, min_size, max_share, min_ari):
    """Select a reliable partition from candidate counts and reusable client subsamples."""
    views = {name: _normalize(distance, quantiles) for name, distance in views.items()}
    names = list(views)
    max_count = max(counts)

    def silhouette(distance, labels):
        if 1 < len(np.unique(labels)) < len(labels):
            return silhouette_score(distance, labels, metric='precomputed')
        return 0.0

    def assess(distance, count):
        labels = np.unique(cluster(distance, count), return_inverse=True)[1]
        sizes = np.bincount(labels)
        sil = silhouette(distance, labels)
        ari = np.mean([adjusted_rand_score(labels[idx], cluster(distance[np.ix_(idx, idx)], count))
                       for idx in resamples])
        imbalance = (count * sizes.max() / len(labels) - 1) / (count - 1)
        score = sil - imbalance_weight * imbalance - instability_weight * (1 - ari)
        return labels, sizes, sil, ari, score

    def valid(distance, assessment):
        labels, sizes, sil, ari, _ = assessment
        if (len(sizes) < 2 or sizes.min() < min_size or sizes.max() / len(labels) > max_share
                or ari < min_ari):
            return False
        null = np.array([silhouette(distance, rng.permutation(labels)) for _ in range(permutations)])
        return sil >= null.mean() + significance * null.std()

    candidates = []
    for count in counts:
        individual = [assess(views[name], count) for name in names]
        qualities = np.array([result[-1] for result in individual])
        if len(names) > 1:
            for i, (name, result) in enumerate(zip(names, individual)):
                other = np.mean([d for key, d in views.items() if key != name], axis=0)
                consensus = normalized_mutual_info_score(result[0], cluster(other, count))
                qualities[i] += consensus_weight * consensus
        caps = np.array([view_caps.get(name, 1) for name in names]) if len(names) > 1 else np.ones(1)
        weights = _capped_softmax(qualities / view_temperature, caps)
        distance = _normalize(sum(w * views[name] for w, name in zip(weights, names)), quantiles)
        result = assess(distance, count)
        score = result[-1] - cost_weight * count / max_count
        candidates.append((score, distance, result, individual, qualities))

    _, distance, result, individual, qualities = max(candidates, key=lambda item: item[0])
    if valid(distance, result):
        return result[0], distance
    for i in np.argsort(-qualities):
        name = names[i]
        if name not in ('route', 'uncertainty') and valid(views[name], individual[i]):
            return individual[i][0], views[name]
    return None
