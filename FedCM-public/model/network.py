import torch
from torch import nn


class FedCM(nn.Module):
    """One client's model. Shared modules have identical shapes across clients."""

    def __init__(self, backbone, adapter, private_expert, head, *, projector,
                 generalist, trunk, router, cluster_gate, private_gate, cluster_experts):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.private_expert = private_expert
        self.head = head
        self.shared = nn.ModuleDict(dict(projector=projector, generalist=generalist,
                                         trunk=trunk, router=router,
                                         cluster_gate=cluster_gate, private_gate=private_gate))
        self.cluster_experts = nn.ModuleList(cluster_experts)

    def encode(self, x):
        return self.shared['projector'](self.adapter(self.backbone(x)))

    def forward(self, x, client_embedding, cluster_embedding, mask, temperature):
        u = self.encode(x)
        e = client_embedding.to(u).expand(len(u), -1)
        c = cluster_embedding.to(u).expand(len(u), -1)
        context = self.shared['trunk'](torch.cat((u, e, c), dim=-1))
        base = self.shared['generalist'](u)

        active = mask.to(device=u.device).nonzero(as_tuple=True)[0]
        alpha = u.new_zeros(len(u), len(self.cluster_experts))
        cluster_gate = torch.zeros_like(base)
        cluster_residual = torch.zeros_like(base)
        if active.numel():
            scores = self.shared['router'](context).index_select(1, active)
            weights = (scores / temperature).softmax(dim=1)
            alpha = alpha.index_copy(1, active, weights)
            deltas = torch.stack([self.cluster_experts[q](base) for q in active.tolist()], dim=1)
            mixed = (weights.unsqueeze(-1) * deltas).sum(dim=1)
            cluster_gate, cluster_residual = self.shared['cluster_gate'](context, mixed, temperature)

        private_gate, private_residual = self.shared['private_gate'](
            context, self.private_expert(u), temperature)
        logits = self.head(base + cluster_residual + private_residual)
        return logits, alpha, cluster_gate, private_gate
