"""Short summary script for IOI outputs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from mechinterp.core.metrics import summarize_logit_diffs
from mechinterp.core.runner import load_experiment_config, read_json, run_dir


def run(task_name: str, config_path: str) -> str:
    """Summarize saved behavior and patch outputs."""

    config = load_experiment_config(config_path)
    root = run_dir(config, config_path)
    behavior_payload = read_json(root / "behavior" / "results.json")

    lines = [f"Task: {task_name}", f"Model: {config.model_name}"]
    for split, split_payload in behavior_payload["splits"].items():
        summary = split_payload["clean_summary"]
        classification = split_payload["classification_summary"]
        lines.append(
            (
                f"{split}: n={summary['count']}, "
                f"mean_logit_diff={summary['mean_logit_diff']:.3f}, "
                f"preference_rate={summary['preference_rate']:.2%}"
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

    patch_path = root / "patch" / "results.json"
    if patch_path.exists():
        patch_payload = read_json(patch_path)
        by_layer: dict[int, list[float]] = defaultdict(list)
        for row in patch_payload["results"]:
            by_layer[int(row["layer"])].append(float(row["normalized_effect"]))
        if by_layer:
            best_layer, best_values = max(by_layer.items(), key=lambda item: mean(item[1]))
            lines.append(f"patch_pairs={patch_payload['pair_count']}")
            lines.append(f"top_layer=L{best_layer} mean_normalized_effect={mean(best_values):.3f}")
    else:
        lines.append("patch results not found")

    summary = "\n".join(lines)
    print(summary)
    return summary
