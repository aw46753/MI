"""Ablation experiment entrypoint."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from mechinterp.analysis.ablation import (
    attention_value_hook_name,
    make_head_ablation_hook,
    make_mlp_ablation_hook,
    mlp_out_hook_name,
)
from mechinterp.core.metrics import final_token_logit_diff
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, log_progress, read_json, run_dir, write_csv, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows, bucket_rows
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.tasks.ioi.score import validate_single_token_candidate


def _sample_rows(rows: list[dict[str, Any]], per_bucket: int = 2) -> list[dict[str, Any]]:
    buckets = bucket_rows(rows)
    selected: list[dict[str, Any]] = []
    for bucket in ("TP", "TN", "FP", "FN"):
        selected.extend(buckets[bucket][:per_bucket])
    return selected


def run(task_name: str, config_path: str, *, device: str | None = None) -> dict[str, Any]:
    """Run attention-head and MLP ablations on sampled examples."""

    config = load_experiment_config(config_path, device=device)
    task = get_task(task_name)
    if not task.supports_ablation():
        raise ValueError(f"Ablation is not implemented for task '{task_name}' yet.")
    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path, device=device)

    rows = annotate_prediction_rows(list(behavior_payload["all_results"]))
    sampled_rows = _sample_rows(rows)
    model = ModelWrapper(config)
    num_layers = int(model.model.cfg.n_layers)
    num_heads = int(getattr(model.model.cfg, "n_heads", 0))
    log_progress(
        f"[ablate] task={task_name} examples={len(sampled_rows)} layers={num_layers} heads={num_heads}"
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(sampled_rows, start=1):
        log_progress(f"[ablate] example {index}/{len(sampled_rows)} split={row['split']} error={row['error_type']}")
        correct_token_id = validate_single_token_candidate(model, row["correct_token"])
        wrong_token_id = validate_single_token_candidate(model, row["wrong_token"])
        base_margin = float(row["margin"])

        for layer in range(num_layers):
            mlp_logits = model.run_with_hooks(
                row["prompt"],
                fwd_hooks=[(mlp_out_hook_name(layer), make_mlp_ablation_hook())],
                return_type="logits",
                prepend_bos=True,
            )
            mlp_margin = final_token_logit_diff(mlp_logits, correct_token_id, wrong_token_id)
            records.append(
                {
                    "prompt": row["prompt"],
                    "split": row["split"],
                    "error_type": row["error_type"],
                    "component_type": "mlp",
                    "layer": layer,
                    "head": None,
                    "base_margin": base_margin,
                    "ablated_margin": mlp_margin,
                    "delta_margin": mlp_margin - base_margin,
                }
            )

            for head in range(num_heads):
                ablated_logits = model.run_with_hooks(
                    row["prompt"],
                    fwd_hooks=[(attention_value_hook_name(layer), make_head_ablation_hook(head))],
                    return_type="logits",
                    prepend_bos=True,
                )
                ablated_margin = final_token_logit_diff(ablated_logits, correct_token_id, wrong_token_id)
                records.append(
                    {
                        "prompt": row["prompt"],
                        "split": row["split"],
                        "error_type": row["error_type"],
                        "component_type": "head",
                        "layer": layer,
                        "head": head,
                        "base_margin": base_margin,
                        "ablated_margin": ablated_margin,
                        "delta_margin": ablated_margin - base_margin,
                    }
                )

    by_component: dict[str, list[float]] = defaultdict(list)
    for record in records:
        key = f"{record['component_type']}:L{record['layer']}"
        by_component[key].append(float(record["delta_margin"]))

    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "num_examples": len(sampled_rows),
        "summary": {component: mean(values) for component, values in by_component.items()},
        "results": records,
    }

    ablation_dir = ensure_dir(run_dir(config, config_path, task_name=task_name) / "ablation")
    write_csv(ablation_dir / "results.csv", records)
    write_json(ablation_dir / "results.json", payload)
    log_progress(f"[ablate] wrote {ablation_dir / 'results.json'}")
    return payload
