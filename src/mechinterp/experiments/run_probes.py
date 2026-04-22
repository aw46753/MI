"""Layerwise probe experiment entrypoint."""

from __future__ import annotations

from typing import Any

from mechinterp.analysis.activations import extract_layerwise_hidden_states
from mechinterp.analysis.probes import train_layerwise_probes
from mechinterp.core.cache import save_activation_artifacts
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, load_experiment_config, read_json, run_dir, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows
from mechinterp.experiments.run_behavior import run as run_behavior


def run(task_name: str, config_path: str) -> dict[str, Any]:
    """Extract hidden states and fit linear probes per layer."""

    config = load_experiment_config(config_path)
    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path)

    rows = annotate_prediction_rows(list(behavior_payload["all_results"]))
    model = ModelWrapper(config)
    selected_rows, layerwise_hidden_states = extract_layerwise_hidden_states(
        model,
        rows,
        max_examples=min(len(rows), 64),
    )
    probe_payload = train_layerwise_probes(layerwise_hidden_states, selected_rows, seed=config.seed)
    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "num_examples": len(selected_rows),
        "probe_results": probe_payload,
    }

    probe_dir = ensure_dir(run_dir(config, config_path, task_name=task_name) / "probes")
    save_activation_artifacts(
        probe_dir / "hidden_states.pt",
        probe_dir / "hidden_state_metadata.json",
        activations={f"layer_{layer}": tensor for layer, tensor in layerwise_hidden_states.items()},
        metadata={"rows": selected_rows},
    )
    write_json(probe_dir / "results.json", payload)
    return payload
