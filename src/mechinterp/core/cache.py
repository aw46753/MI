"""Activation cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def build_names_filter(hook_names: Iterable[str]) -> Callable[[str], bool]:
    """Build a names_filter matching exact names or hook suffixes."""

    allowed = tuple(hook_names)

    def names_filter(name: str) -> bool:
        return any(name == item or name.endswith(item) for item in allowed)

    return names_filter


def compact_cache(cache: Any) -> dict[str, Any]:
    """Detach and move cached tensors to CPU for saving."""

    compact: dict[str, Any] = {}
    for key, value in cache.items():
        tensor = value
        try:
            tensor = value.detach().cpu()
        except AttributeError:
            pass
        compact[str(key)] = tensor
    return compact


def select_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a stable prefix of records for caching."""
    return list(records[:limit])


def save_activation_artifacts(
    activations_path: str | Path,
    metadata_path: str | Path,
    activations: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Save cached activations and JSON metadata."""

    import torch

    activations_path = Path(activations_path)
    metadata_path = Path(metadata_path)
    activations_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(activations, activations_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
