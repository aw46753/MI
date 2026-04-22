"""Layerwise activation extraction helpers."""

from __future__ import annotations

from typing import Any

from mechinterp.core.hooks import resid_pre_hook_name


def resid_hook_names(num_layers: int) -> list[str]:
    """Return residual-stream input hook names for all layers."""

    return [resid_pre_hook_name(layer) for layer in range(num_layers)]


def extract_layerwise_hidden_states(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    max_examples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """Extract final-token residual states for each example and layer."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Torch is required for hidden-state extraction.") from exc

    selected_rows = rows[: max_examples] if max_examples is not None else list(rows)
    if not selected_rows:
        return [], {}

    num_layers = int(model.model.cfg.n_layers)
    hook_names = resid_hook_names(num_layers)
    names_filter = lambda name: name in hook_names  # noqa: E731

    per_layer: dict[int, list[Any]] = {layer: [] for layer in range(num_layers)}
    for row in selected_rows:
        _, cache = model.run_with_cache(
            row["prompt"],
            names_filter=names_filter,
            return_type="logits",
            prepend_bos=True,
        )
        for layer, hook_name in enumerate(hook_names):
            tensor = cache[hook_name][:, -1, :].detach().cpu()
            per_layer[layer].append(tensor.squeeze(0))

    layerwise_hidden_states = {
        layer: torch.stack(states, dim=0) for layer, states in per_layer.items() if states
    }
    return selected_rows, layerwise_hidden_states

