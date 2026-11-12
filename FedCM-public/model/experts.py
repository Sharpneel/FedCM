from torch import nn


class LowRankExpert(nn.Module):
    def __init__(self, width, rank, init_std):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width)
        nn.init.normal_(self.down.weight, std=width ** -0.5)
        nn.init.normal_(self.up.weight, std=init_std)
        nn.init.normal_(self.up.bias, std=init_std)

    def forward(self, base):
        return self.up(self.down(self.norm(base)))


class ResidualGate(nn.Module):
    def __init__(self, network, width, floor):
        super().__init__()
        self.network = network
        self.norm = nn.LayerNorm(width)
        self.floor = floor

    def forward(self, context, residual, temperature):
        gate = self.floor + (1 - self.floor) * (self.network(context) / temperature).sigmoid()
        return gate, gate * self.norm(residual)
