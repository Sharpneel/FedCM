import torch
from torch.utils.data import DataLoader, TensorDataset


def make_batch(group, groups, samples):
    """Add a group-specific feature shift and a shared binary prediction task."""
    labels = torch.arange(samples) % 2
    features = 0.13 * torch.randn(samples, groups + 1)
    features[:, group] += 3
    features[:, -1] += 2 * labels - 1
    return features, labels


def synthetic_loaders(domains, groups):
    train = [make_batch(group, groups, samples=28) for group in domains]
    test = [make_batch(group, groups, samples=14) for group in domains]
    loaders = [DataLoader(TensorDataset(*batch), batch_size=7, shuffle=True) for batch in train]
    test_loaders = [DataLoader(TensorDataset(*batch), batch_size=len(batch[1])) for batch in test]
    return loaders, test_loaders
