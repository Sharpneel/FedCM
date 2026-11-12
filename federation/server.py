import torch


def _average(entries):
    total = sum(weight for weight, _ in entries)
    return {
        key: (sum((weight / total) * state[key] for weight, state in entries)
              if torch.is_floating_point(value) else value)
        for key, value in entries[-1][1].items()
    }


def aggregate(server, updates, active_experts, *, confidence_floor, weight_global_by_confidence):
    """Aggregate global modules by client and each cluster expert by membership."""
    weighted = [(update, max(update.confidence, confidence_floor)) for update in updates]
    server.shared.load_state_dict(_average([
        (update.samples * (confidence if weight_global_by_confidence else 1),
         update.state.global_modules)
        for update, confidence in weighted
    ]))
    for q in active_experts:
        entries = []
        for update, confidence in weighted:
            weight = update.samples * update.membership[q] * confidence
            if q in update.state.experts and weight > 0:
                entries.append((weight, update.state.experts[q]))
        if entries:
            server.cluster_experts[q].load_state_dict(_average(entries))
