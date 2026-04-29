"""Plot generation entrypoint."""

from __future__ import annotations

from pathlib import Path

from mechinterp.core.runner import ensure_dir, load_experiment_config, read_json, run_dir
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.plots.generate import (
    plot_ablation_head_heatmap,
    plot_ablation_mlp_heatmap,
    plot_confusion_matrix,
    plot_fpr_fnr_by_split,
    plot_margin_histograms,
    plot_patching_heatmap,
    plot_probe_accuracy,
)


def run(task_name: str, config_path: str, *, device: str | None = None) -> dict[str, str]:
    """Generate plots from saved experiment outputs."""

    config = load_experiment_config(config_path, device=device)
    root = run_dir(config, config_path, task_name=task_name)
    behavior_path = root / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path, device=device)

    plot_dir = ensure_dir(root / "plots")
    outputs: dict[str, str] = {}

    confusion_path = plot_dir / "confusion_matrix.png"
    plot_confusion_matrix(behavior_payload["classification_summary"], confusion_path)
    outputs["confusion_matrix"] = str(confusion_path)

    subgroup_path = plot_dir / "fpr_fnr_by_split.png"
    plot_fpr_fnr_by_split(behavior_payload["splits"], subgroup_path)
    outputs["fpr_fnr_by_split"] = str(subgroup_path)

    margin_path = plot_dir / "margin_histograms.png"
    plot_margin_histograms(list(behavior_payload["all_results"]), margin_path)
    outputs["margin_histograms"] = str(margin_path)

    probe_path = root / "probes" / "results.json"
    if probe_path.exists():
        probe_payload = read_json(probe_path)
        probe_plot_path = plot_dir / "probe_accuracy.png"
        plot_probe_accuracy(probe_payload, probe_plot_path)
        outputs["probe_accuracy"] = str(probe_plot_path)

    patch_path = root / "patch" / "results.json"
    if patch_path.exists():
        patch_payload = read_json(patch_path)
        heatmap_path = plot_dir / "patching_heatmap.png"
        plot_patching_heatmap(patch_payload.get("aggregate", {}), heatmap_path)
        outputs["patching_heatmap"] = str(heatmap_path)

    ablation_path = root / "ablation" / "results.json"
    if ablation_path.exists():
        ablation_payload = read_json(ablation_path)
        try:
            head_heatmap_path = plot_dir / "ablation_head_heatmap.html"
            plot_ablation_head_heatmap(list(ablation_payload.get("results", [])), head_heatmap_path)
            outputs["ablation_head_heatmap"] = str(head_heatmap_path)

            mlp_heatmap_path = plot_dir / "ablation_mlp_heatmap.html"
            plot_ablation_mlp_heatmap(list(ablation_payload.get("results", [])), mlp_heatmap_path)
            outputs["ablation_mlp_heatmap"] = str(mlp_heatmap_path)
        except RuntimeError:
            pass

    return outputs
