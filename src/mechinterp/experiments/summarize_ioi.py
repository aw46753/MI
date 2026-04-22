"""Short summary script for task outputs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from mechinterp.core.runner import load_experiment_config, read_json, run_dir


def run(task_name: str, config_path: str) -> str:
    """Summarize saved behavior and patch outputs."""

    config = load_experiment_config(config_path)
    root = run_dir(config, config_path, task_name=task_name)
    behavior_payload = read_json(root / "behavior" / "results.json")

    lines = [f"Task: {task_name}", f"Model: {config.model_name}"]
    for split, split_payload in behavior_payload["splits"].items():
        summary = split_payload["classification_summary"]
        classification = split_payload["classification_summary"]
        lines.append(
            (
                f"{split}: n={summary['count']}, "
                f"accuracy={summary['accuracy']:.2%}, "
                f"precision={summary['precision']:.2%}, "
                f"recall={summary['recall']:.2%}"
            )
        )
        lines.append(
            (
                f"{split}: FNR={classification['fnr']:.2%}, "
                f"FPR={classification['fpr']:.2%}, "
                f"TPR={classification['tpr']:.2%}, "
                f"TNR={classification['tnr']:.2%}"
            )
        )

    analysis_path = root / "analysis" / "results.json"
    if analysis_path.exists():
        analysis_payload = read_json(analysis_path)
        overall_buckets = analysis_payload["overall"]["bucket_summary"]
        lines.append(
            "buckets: "
            + ", ".join(f"{bucket}={overall_buckets[bucket]['count']}" for bucket in ("TP", "TN", "FP", "FN"))
        )

    probe_path = root / "probes" / "results.json"
    if probe_path.exists():
        probe_payload = read_json(probe_path)
        layer_scores = probe_payload["probe_results"]["layers"]
        if layer_scores:
            best_layer = max(
                layer_scores,
                key=lambda layer: layer_scores[layer]["all_examples"]["accuracy"],
            )
            lines.append(
                f"best_probe_layer=L{best_layer} accuracy={layer_scores[best_layer]['all_examples']['accuracy']:.2%}"
            )

    patch_path = root / "patch" / "results.json"
    if patch_path.exists():
        patch_payload = read_json(patch_path)
        by_layer: dict[int, list[float]] = defaultdict(list)
        for row in patch_payload["results"]:
            by_layer[int(row["layer"])].append(float(row["normalized_effect"]))
        if by_layer:
            best_layer, best_values = max(by_layer.items(), key=lambda item: mean(item[1]))
            lines.append(
                f"patch_pairs={patch_payload['pair_count']} position_mode={patch_payload.get('position_mode', 'final')}"
            )
            lines.append(f"top_layer=L{best_layer} mean_normalized_effect={mean(best_values):.3f}")

        top_entries = patch_payload.get("aggregate", {}).get("top_entries", [])
        if top_entries:
            top_entry = top_entries[0]
            lines.append(
                (
                    f"top_site=L{top_entry['layer']} P{top_entry['position']} "
                    f"mean_normalized_effect={top_entry['mean_normalized_effect']:.3f}"
                )
            )
    else:
        lines.append("patch results not found")

    summary = "\n".join(lines)
    print(summary)
    return summary
