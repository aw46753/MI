"""Shared orchestration helpers for experiments and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mechinterp.core.config import ExperimentConfig, load_config, override_device
from mechinterp.tasks.addition import AdditionTask
from mechinterp.tasks.bigvul import BigVulTask
from mechinterp.tasks.greater_than import GreaterThanTask
from mechinterp.tasks.ioi import IOITask
from mechinterp.tasks.sva import SVATask


TASK_REGISTRY = {
    "addition": AdditionTask(),
    "bigvul": BigVulTask(),
    "greater_than": GreaterThanTask(),
    "ioi": IOITask(),
    "sva": SVATask(),
}


def get_task(task_name: str) -> Any:
    """Look up a registered task."""
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {sorted(TASK_REGISTRY)}") from exc


def load_experiment_config(config_path: str, *, device: str | None = None) -> ExperimentConfig:
    """Load YAML config and attach a stable run name based on file stem."""

    config = load_config(config_path)
    return override_device(config, device)


def run_dir(config: ExperimentConfig, config_path: str, *, task_name: str | None = None) -> Path:
    """Return the root output directory for a config file."""

    stem = Path(config_path).stem
    if task_name is None:
        return Path(config.output.output_dir) / stem
    return Path(config.output.output_dir) / task_name / stem


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_progress(message: str) -> None:
    """Print a short progress update for long-running commands."""

    print(message, flush=True)


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with indentation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Read JSON payload from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
