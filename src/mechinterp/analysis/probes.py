"""Simple linear probes over layerwise hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProbeResult:
    """Summary of one linear-probe run."""

    accuracy: float
    count: int

    def to_dict(self) -> dict[str, float]:
        return {"accuracy": self.accuracy, "count": self.count}


def _fit_binary_probe(features: Any, labels: Any, *, seed: int) -> Any:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    probe = nn.Linear(features.shape[-1], 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    inputs = features.float()
    targets = labels.float().unsqueeze(-1)
    for _ in range(200):
        optimizer.zero_grad()
        logits = probe(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
    return probe


def _evaluate_binary_probe(probe: Any, features: Any, labels: Any) -> ProbeResult:
    import torch

    if int(labels.shape[0]) == 0:
        return ProbeResult(accuracy=0.0, count=0)

    with torch.no_grad():
        logits = probe(features.float()).squeeze(-1)
        predictions = (logits > 0).long()
        accuracy = float((predictions == labels.long()).float().mean().item())
    return ProbeResult(accuracy=accuracy, count=int(labels.shape[0]))


def train_layerwise_probes(
    layerwise_hidden_states: dict[int, Any],
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Train class and error-bucket probes for each layer."""

    import torch

    if not rows:
        return {"layers": {}}

    class_labels = torch.tensor([int(row["gold_label"]) for row in rows], dtype=torch.long)
    positive_error_mask = torch.tensor(
        [int(row["error_type"] in {"TP", "FN"}) for row in rows], dtype=torch.bool
    )
    positive_error_labels = torch.tensor(
        [1 if row["error_type"] == "TP" else 0 for row in rows], dtype=torch.long
    )
    negative_error_mask = torch.tensor(
        [int(row["error_type"] in {"TN", "FP"}) for row in rows], dtype=torch.bool
    )
    negative_error_labels = torch.tensor(
        [1 if row["error_type"] == "TN" else 0 for row in rows], dtype=torch.long
    )

    payload: dict[str, Any] = {"layers": {}}
    for layer, features in layerwise_hidden_states.items():
        layer_payload: dict[str, Any] = {}

        class_probe = _fit_binary_probe(features, class_labels, seed=seed + layer)
        layer_payload["all_examples"] = _evaluate_binary_probe(class_probe, features, class_labels).to_dict()

        positive_features = features[positive_error_mask]
        positive_labels = positive_error_labels[positive_error_mask]
        if int(positive_labels.shape[0]) >= 2 and len(set(positive_labels.tolist())) == 2:
            positive_probe = _fit_binary_probe(positive_features, positive_labels, seed=seed + 100 + layer)
            layer_payload["fn_vs_tp"] = _evaluate_binary_probe(
                positive_probe,
                positive_features,
                positive_labels,
            ).to_dict()
        else:
            layer_payload["fn_vs_tp"] = ProbeResult(accuracy=0.0, count=int(positive_labels.shape[0])).to_dict()

        negative_features = features[negative_error_mask]
        negative_labels = negative_error_labels[negative_error_mask]
        if int(negative_labels.shape[0]) >= 2 and len(set(negative_labels.tolist())) == 2:
            negative_probe = _fit_binary_probe(negative_features, negative_labels, seed=seed + 200 + layer)
            layer_payload["fp_vs_tn"] = _evaluate_binary_probe(
                negative_probe,
                negative_features,
                negative_labels,
            ).to_dict()
        else:
            layer_payload["fp_vs_tn"] = ProbeResult(accuracy=0.0, count=int(negative_labels.shape[0])).to_dict()

        payload["layers"][str(layer)] = layer_payload
    return payload

