"""Attention-head and MLP ablation helpers."""

from __future__ import annotations

from typing import Any, Callable


def _get_act_name(act_name: str, layer: int) -> str:
    from transformer_lens import utils

    return str(utils.get_act_name(act_name, layer))


def make_head_ablation_hook(head_index: int) -> Callable[[Any, Any], Any]:
    """Return a hook that zeros one attention head value stream."""

    def hook_fn(value: Any, hook: Any) -> Any:
        patched = value.clone()
        patched[:, :, head_index, :] = 0.0
        return patched

    return hook_fn


def make_mlp_ablation_hook() -> Callable[[Any, Any], Any]:
    """Return a hook that zeros an MLP output."""

    def hook_fn(mlp_out: Any, hook: Any) -> Any:
        return mlp_out.clone().zero_()

    return hook_fn


def attention_value_hook_name(layer: int) -> str:
    """Return the attention-value activation hook name."""

    return _get_act_name("v", layer)


def mlp_out_hook_name(layer: int) -> str:
    """Return the MLP output activation hook name."""

    return _get_act_name("mlp_out", layer)
