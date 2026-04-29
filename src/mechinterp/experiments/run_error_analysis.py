"""Error-bucket and matched-pair analysis entrypoint."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from mechinterp.analysis.activations import resid_hook_names
from mechinterp.analysis.error_buckets import analyze_error_buckets, compare_bucket_pair
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, log_progress, read_json, run_dir, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.core.model import ModelWrapper


def _activation_difference_summary(model: Any, pairs: list[Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Torch is required for activation-difference analysis.") from exc

    if not pairs:
        return {"pair_count": 0, "layer_mean_l2": {}}

    num_layers = int(model.model.cfg.n_layers)
    hook_names = resid_hook_names(num_layers)
    names_filter = lambda name: name in hook_names  # noqa: E731

    layer_differences: dict[int, list[float]] = defaultdict(list)
    for pair in pairs:
        _, source_cache = model.run_with_cache(pair.source_prompt, names_filter=names_filter, return_type="logits")
        _, target_cache = model.run_with_cache(pair.target_prompt, names_filter=names_filter, return_type="logits")
        for layer, hook_name in enumerate(hook_names):
            source_tensor = source_cache[hook_name][:, -1, :].detach().cpu()
            target_tensor = target_cache[hook_name][:, -1, :].detach().cpu()
            distance = float(torch.norm(source_tensor - target_tensor, dim=-1).mean().item())
            layer_differences[layer].append(distance)

    return {
        "pair_count": len(pairs),
        "layer_mean_l2": {
            str(layer): mean(values) for layer, values in layer_differences.items()
        },
    }


def run(task_name: str, config_path: str) -> dict[str, Any]:
    """Run error bucket and matched-pair analysis."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    if not task.supports_analysis():
        raise ValueError(f"Analysis is not implemented for task '{task_name}' yet.")
    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path)

    rows = annotate_prediction_rows(list(behavior_payload["all_results"]))
    model = ModelWrapper(config)
    log_progress(f"[analyze] task={task_name} rows={len(rows)}")

    pair_payload: dict[str, Any] = {}
    for source_error_type, target_error_type in (("FN", "TP"), ("FP", "TN")):
        log_progress(f"[analyze] building pairs {source_error_type}->{target_error_type}")
        pairs = task.build_error_pairs(rows, source_error_type, target_error_type, model)
        pair_key = f"{source_error_type}_to_{target_error_type}"
        pair_payload[pair_key] = {
            "pair_count": len(pairs),
            "pairs": [pair.to_dict() for pair in pairs[:20]],
            "activation_differences": _activation_difference_summary(model, pairs[:10]),
        }
        log_progress(f"[analyze] {pair_key} pair_count={len(pairs)}")

    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "overall": analyze_error_buckets(rows),
        "bucket_comparisons": {
            "fp_vs_tn": compare_bucket_pair(rows, "FP", "TN"),
            "fn_vs_tp": compare_bucket_pair(rows, "FN", "TP"),
        },
        "matched_pairs": pair_payload,
    }
    analysis_dir = ensure_dir(run_dir(config, config_path, task_name=task_name) / "analysis")
    write_json(analysis_dir / "results.json", payload)
    log_progress(f"[analyze] wrote {analysis_dir / 'results.json'}")
    return payload
