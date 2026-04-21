"""Shared hook helpers."""

from __future__ import annotations

from typing import Any, Callable


def _candidate_cache_names(hook_name: str) -> list[str]:
    """Return compatible cache names for a residual hook."""

    candidates = [hook_name]
    if hook_name.endswith(".hook_in"):
        candidates.append(hook_name.replace(".hook_in", ".hook_resid_pre"))
    if hook_name.endswith(".hook_resid_pre"):
        candidates.append(hook_name.replace(".hook_resid_pre", ".hook_in"))
    return candidates


def make_residual_patch_hook(clean_cache: Any, position: int) -> Callable[[Any, Any], Any]:
    """Return a hook that patches the residual stream at one position."""

    def patch_hook(corrupted_activation: Any, hook: Any) -> Any:
        clean_value = None
        for cache_name in _candidate_cache_names(hook.name):
            try:
                clean_value = clean_cache[cache_name][:, position, :].clone()
                break
            except KeyError:
                continue

        if clean_value is None:
            raise KeyError(
                f"Could not find a cached clean activation for hook {hook.name!r}. "
                f"Tried {_candidate_cache_names(hook.name)}."
            )

        patched = corrupted_activation.clone()
        patched[:, position : position + 1, :] = clean_value.unsqueeze(1)
        return patched

    return patch_hook


def resid_pre_hook_name(layer: int) -> str:
    """Return the residual-stream input hook name used by TransformerBridge."""
    return f"blocks.{layer}.hook_in"
