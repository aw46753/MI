"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CompatibilityModeConfig:
    """Compatibility flags applied after TransformerBridge boot."""

    center_unembed: bool = True
    center_writing_weights: bool = True
    fold_ln: bool = True
    refactor_factored_attn_matrices: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-related settings."""

    dataset_sizes: dict[str, int] = field(default_factory=dict)
    dataset_path: str | None = None
    target_cwes: list[str] = field(default_factory=list)
    pairs_per_cwe: int | None = None
    split_mode: str = "standard_shifted"
    names: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    shifted_name_count: int = 0
    shifted_template_count: int = 0


@dataclass(frozen=True)
class CacheConfig:
    """Activation cache settings."""

    cache_hook_names: list[str] = field(default_factory=list)
    stop_at_layer: int | None = None
    cache_num_examples: int = 4


@dataclass(frozen=True)
class PatchConfig:
    """Patching sweep settings."""

    max_layer: int = 0
    position_mode: str = "final"
    max_pairs: int | None = None


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem output settings."""

    output_dir: str = "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Root config shared across experiments."""

    model_name: str
    device: str
    seed: int
    dataset: DatasetConfig
    cache: CacheConfig
    patch: PatchConfig
    output: OutputConfig
    compatibility_mode: CompatibilityModeConfig = field(default_factory=CompatibilityModeConfig)

    @property
    def run_name(self) -> str:
        """Return a stable run directory name."""
        return "run"


def _require_keys(data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def _validate_dataset_config(dataset: DatasetConfig) -> None:
    if dataset.target_cwes:
        if not dataset.dataset_path:
            raise ValueError("dataset_path is required for Big-Vul datasets")
        if dataset.pairs_per_cwe is None or dataset.pairs_per_cwe <= 0:
            raise ValueError("pairs_per_cwe must be a positive integer")
        if dataset.split_mode != "by_cwe":
            raise ValueError("split_mode must be 'by_cwe' for Big-Vul datasets")
        return

    if "standard" not in dataset.dataset_sizes or "shifted" not in dataset.dataset_sizes:
        raise ValueError("dataset_sizes must define both 'standard' and 'shifted'")

    if dataset.dataset_sizes["standard"] <= 0 or dataset.dataset_sizes["shifted"] <= 0:
        raise ValueError("dataset_sizes values must be positive")


def _validate_ioi_dataset_config(dataset: DatasetConfig) -> None:
    if len(dataset.names) < 4:
        raise ValueError("At least four names are required for IOI generation")

    if len(dataset.templates) < 1:
        raise ValueError("At least one template id is required")

    if dataset.shifted_name_count < 2:
        raise ValueError("shifted_name_count must be at least 2")

    if dataset.shifted_name_count >= len(dataset.names):
        raise ValueError("shifted_name_count must be smaller than the number of names")

    if dataset.shifted_template_count < 1:
        raise ValueError("shifted_template_count must be at least 1")

    if dataset.shifted_template_count >= len(dataset.templates):
        raise ValueError("shifted_template_count must be smaller than the number of template ids")


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a top-level mapping")

    _require_keys(
        raw,
        [
            "model_name",
            "device",
            "seed",
            "cache_hook_names",
            "max_layer",
            "output_dir",
        ],
    )

    dataset = DatasetConfig(
        dataset_sizes=dict(raw.get("dataset_sizes", {})),
        dataset_path=raw.get("dataset_path"),
        target_cwes=list(raw.get("target_cwes", [])),
        pairs_per_cwe=int(raw["pairs_per_cwe"]) if raw.get("pairs_per_cwe") is not None else None,
        split_mode=str(raw.get("split_mode", "standard_shifted")),
        names=list(raw.get("names", [])),
        templates=list(raw.get("templates", [])),
        shifted_name_count=int(raw.get("shifted_name_count", 2)),
        shifted_template_count=int(raw.get("shifted_template_count", 1)),
    )
    _validate_dataset_config(dataset)
    if any(
        key in raw
        for key in ("names", "templates", "shifted_name_count", "shifted_template_count")
    ):
        _validate_ioi_dataset_config(dataset)

    cache = CacheConfig(
        cache_hook_names=list(raw.get("cache_hook_names", [])),
        stop_at_layer=raw.get("stop_at_layer"),
        cache_num_examples=int(raw.get("cache_num_examples", 4)),
    )
    patch = PatchConfig(
        max_layer=int(raw.get("max_layer", 0)),
        position_mode=str(raw.get("patch_position_mode", "final")),
        max_pairs=int(raw["patch_max_pairs"]) if raw.get("patch_max_pairs") is not None else None,
    )
    if patch.position_mode not in {"all", "final"}:
        raise ValueError("patch_position_mode must be either 'all' or 'final'")
    if patch.max_pairs is not None and patch.max_pairs <= 0:
        raise ValueError("patch_max_pairs must be a positive integer when provided")
    output = OutputConfig(output_dir=str(raw["output_dir"]))
    compatibility_mode = CompatibilityModeConfig(**dict(raw.get("compatibility_mode", {})))

    return ExperimentConfig(
        model_name=str(raw["model_name"]),
        device=str(raw["device"]),
        seed=int(raw["seed"]),
        dataset=dataset,
        cache=cache,
        patch=patch,
        output=output,
        compatibility_mode=compatibility_mode,
    )
