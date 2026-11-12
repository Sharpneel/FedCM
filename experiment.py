import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from torch.nn.functional import normalize

if __package__:
    from .clustering import distance_views, fit_embeddings, routing_context, select_partition
    from .clustering.signatures import client_signatures
    from .datasets.cifar10 import cifar10_loaders
    from .datasets.synthetic import synthetic_loaders
    from .federation.rounds import run_round, test_accuracy
    from .model.backbones import cifar10_backbone
    from .model.builder import make_model
else:
    from clustering import distance_views, fit_embeddings, routing_context, select_partition
    from clustering.signatures import client_signatures
    from datasets.cifar10 import cifar10_loaders
    from datasets.synthetic import synthetic_loaders
    from federation.rounds import run_round, test_accuracy
    from model.backbones import cifar10_backbone
    from model.builder import make_model


def group_clients(signatures, experts, embedding_dim, rng, routing_options):
    views = distance_views(signatures, use_feedback=False)

    def cluster(distance, count):
        return AgglomerativeClustering(n_clusters=count, metric='precomputed',
                                       linkage='average').fit_predict(distance)

    n = len(signatures)
    resamples = [rng.choice(n, n - 2, replace=False) for _ in range(7)]
    partition = select_partition(
        views, cluster, range(2, experts + 1), resamples, rng,
        quantiles=routing_options['quantiles'], view_temperature=0.37, view_caps={'label': 0.18},
        imbalance_weight=0.21, instability_weight=0.34, consensus_weight=0.12,
        cost_weight=0.04, permutations=17, significance=1.4, min_size=2,
        max_share=0.57, min_ari=0.65,
    )
    embeddings = np.zeros((n, embedding_dim))
    if partition is not None:
        labels, distance = partition
        moments = torch.from_numpy(np.stack([signature.mean for signature in signatures]))
        vectors = normalize(moments - moments.mean(dim=0), dim=1)
        projector = nn.Linear(vectors.size(1), embedding_dim)
        optimizer = torch.optim.SGD(projector.parameters(), lr=0.065)
        embeddings = fit_embeddings(projector, vectors, labels, distance, optimizer, 6,
                                    neighbors=2, hard_negatives=3, refresh_every=3,
                                    temperature=0.71)
    return routing_context(partition, embeddings, experts, **routing_options)


def run_experiment(data_dir, device, synthetic):
    torch.set_num_threads(1 if synthetic else 4)
    seed = 137
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # These settings are for the standalone examples.
    groups, embedding_dim = 3, 5
    if synthetic:
        domains = np.repeat(np.arange(groups), 3)
        loaders, test_loaders = synthetic_loaders(domains, groups)
        backbone_widths = (5, 7)
        backbones = [(nn.Sequential(nn.Linear(groups + 1, w), nn.Tanh()), w)
                     for w in (backbone_widths[i % 2] for i in range(len(domains)))]
        reference_features = nn.Identity()
        num_classes, width, rounds, learning_rate = 2, 9, 2, 0.075
    else:
        domains = np.repeat(np.arange(groups), 4)
        loaders, test_loaders = cifar10_loaders(data_dir, domains, rng)
        backbones = [cifar10_backbone(i % 2) for i in range(len(domains))]
        reference_features = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten())
        num_classes, width, rounds, learning_rate = 10, 32, 4, 0.05

    clients = [make_model(backbone, size, num_classes, width, embedding_dim, groups).to(device)
               for backbone, size in backbones]
    server = make_model(nn.Identity(), width, num_classes, width, embedding_dim, groups)
    optimizers = [torch.optim.SGD(client.parameters(), lr=learning_rate) for client in clients]
    temperature = 0.83
    routing_options = dict(quantiles=(0.10, 0.90), temperature=0.68, neighbors=2,
                           smoothing=0.19, neighbor_experts=1, link_weight=0.27)

    embeddings = np.zeros((len(clients), embedding_dim))
    warmup = routing_context(None, embeddings, groups, **routing_options)
    active = run_round(server, clients, loaders, optimizers, warmup, temperature)
    print(f'Round 1/{rounds} (warm-up): {active} active cluster experts', flush=True)

    signatures = client_signatures(loaders, reference_features, num_classes, groups)
    context = group_clients(signatures, groups, embedding_dim, rng, routing_options)
    for round_number in range(2, rounds + 1):
        active = run_round(server, clients, loaders, optimizers, context, temperature)
        print(f'Round {round_number}/{rounds}: {active} active cluster experts', flush=True)
    accuracy = test_accuracy(server, clients, test_loaders, context, temperature)
    dataset = 'Synthetic' if synthetic else 'CIFAR-10'
    print(f'{dataset} test accuracy: {accuracy:.1%}')
