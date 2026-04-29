"""Behavior experiment entrypoint."""

from __future__ import annotations

from mechinterp.analysis.error_buckets import analyze_error_buckets, compare_bucket_pair
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, log_progress, run_dir, write_csv, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows, confusion_metrics


def run(task_name: str, config_path: str) -> dict:
    """Run behavior scoring for a task."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    log_progress(f"[behavior] task={task_name} loading model and scoring splits")
    model = ModelWrapper(config) if task.requires_model() else None

    primary_rows_all: list[dict] = []
    all_rows: list[dict] = []
    splits_payload: dict[str, dict] = {}

    for split in task.split_names(config):
        log_progress(f"[behavior] scoring split={split}")
        behavior_split = task.build_behavior_split(model, split, config)
        primary_rows = annotate_prediction_rows(list(behavior_split["primary_rows"]))
        split_all_rows = annotate_prediction_rows(list(behavior_split["all_rows"]))
        positive_rows = [row for row in split_all_rows if bool(row["expected_positive"])]
        negative_rows = [row for row in split_all_rows if not bool(row["expected_positive"])]

        primary_rows_all.extend(primary_rows)
        all_rows.extend(split_all_rows)

        positive_name = str(behavior_split.get("positive_summary_name", "positive"))
        negative_name = str(behavior_split.get("negative_summary_name", "negative"))
        positive_summary = analyze_error_buckets(positive_rows)
        negative_summary = analyze_error_buckets(negative_rows)
        splits_payload[split] = {
            "results": primary_rows,
            "all_results": split_all_rows,
            "positive_results": positive_rows,
            "negative_results": negative_rows,
            "positive_summary": positive_summary,
            "negative_summary": negative_summary,
            f"{positive_name}_results": positive_rows,
            f"{negative_name}_results": negative_rows,
            f"{positive_name}_summary": positive_summary,
            f"{negative_name}_summary": negative_summary,
            "classification_summary": confusion_metrics(split_all_rows),
            "bucket_comparisons": {
                "fp_vs_tn": compare_bucket_pair(split_all_rows, "FP", "TN"),
                "fn_vs_tp": compare_bucket_pair(split_all_rows, "FN", "TP"),
            },
        }
        log_progress(f"[behavior] finished split={split} examples={len(split_all_rows)}")

    behavior_dir = ensure_dir(run_dir(config, config_path, task_name=task_name) / "behavior")
    write_csv(behavior_dir / "results.csv", all_rows)
    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "results": primary_rows_all,
        "positive_results": [row for row in all_rows if bool(row["expected_positive"])],
        "negative_results": [row for row in all_rows if not bool(row["expected_positive"])],
        "all_results": all_rows,
        "classification_summary": confusion_metrics(all_rows),
        "splits": splits_payload,
    }
    write_json(behavior_dir / "results.json", payload)
    log_progress(f"[behavior] wrote {behavior_dir / 'results.json'}")
    return payload
