"""Layerwise probe experiment entrypoint."""

from __future__ import annotations

from typing import Any

from mechinterp.analysis.activations import extract_layerwise_hidden_states
from mechinterp.analysis.probes import train_layerwise_probes
from mechinterp.core.cache import save_activation_artifacts
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, log_progress, read_json, run_dir, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows
from mechinterp.experiments.run_behavior import run as run_behavior


def run(task_name: str, config_path: str) -> dict[str, Any]:
    """Extract hidden states and fit linear probes per layer."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    if not task.supports_probes():
        raise ValueError(f"Probe experiments are not implemented for task '{task_name}' yet.")
    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path)

    rows = annotate_prediction_rows(list(behavior_payload["all_results"]))
    model = ModelWrapper(config)
    log_progress(f"[probe] task={task_name} extracting hidden states from up to {min(len(rows), 64)} rows")
    selected_rows, layerwise_hidden_states = extract_layerwise_hidden_states(
        model,
        rows,
        max_examples=min(len(rows), 64),
    )
    log_progress(f"[probe] training probes on {len(selected_rows)} examples")
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
    log_progress(f"[probe] wrote {probe_dir / 'results.json'}")
    return payload
