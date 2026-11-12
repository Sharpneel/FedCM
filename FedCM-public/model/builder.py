from torch import nn

from .experts import LowRankExpert, ResidualGate
from .network import FedCM


def make_model(backbone, backbone_width, num_classes, width, embedding_dim, experts):
    return FedCM(
        backbone=backbone,
        adapter=nn.Linear(backbone_width, width),
        private_expert=nn.Linear(width, width),
        head=nn.Linear(width, num_classes),
        projector=nn.Sequential(nn.Linear(width, width), nn.ReLU()),
        generalist=nn.Linear(width, width),
        trunk=nn.Sequential(nn.Linear(width + 2 * embedding_dim, width), nn.ReLU()),
        router=nn.Linear(width, experts),
        cluster_gate=ResidualGate(nn.Linear(width, width), width, floor=0.17),
        private_gate=ResidualGate(nn.Linear(width, width), width, floor=0.13),
        cluster_experts=[LowRankExpert(width, rank=3, init_std=0.018) for _ in range(experts)],
    )
