from dataclasses import dataclass


@dataclass
class SharedState:
    global_modules: dict
    experts: dict


@dataclass
class ClientUpdate:
    state: SharedState
    samples: int
    membership: object
    confidence: float


def shared_state(model, mask=None):
    """Snapshot shared modules and optionally restrict the cluster payload."""
    def snapshot(module):
        return {key: value.detach().to(device='cpu', copy=True)
                for key, value in module.state_dict().items()}

    return SharedState(snapshot(model.shared), {
        q: snapshot(expert) for q, expert in enumerate(model.cluster_experts)
        if mask is None or mask[q]
    })


def load_shared(model, state):
    model.shared.load_state_dict(state.global_modules)
    for q, expert in state.experts.items():
        model.cluster_experts[q].load_state_dict(expert)
