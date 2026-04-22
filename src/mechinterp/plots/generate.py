"""Generate summary figures from saved experiment outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install project dependencies with "
            "`pip install -r requirements.txt` or `pip install -e '.[dev]'`."
        ) from exc

    return plt


def _import_plotly():
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError(
            "Interactive heatmaps require plotly. Install project dependencies with "
            "`pip install -r requirements.txt` or `pip install -e '.[dev]'`."
        ) from exc

    return px


def _write_notebook_imshow(
    matrix: list[list[float | None]],
    output_path: Path,
    *,
    x: list[str] | list[int] | None = None,
    y: list[str] | list[int] | None = None,
    title: str,
    xaxis: str,
    yaxis: str,
) -> None:
    px = _import_plotly()
    figure = px.imshow(
        matrix,
        x=x,
        y=y,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        title=title,
        aspect="auto",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def plot_confusion_matrix(summary: dict[str, Any], output_path: Path) -> None:
    plt = _import_matplotlib()
    matrix = [
        [summary["tp"], summary["fn"]],
        [summary["fp"], summary["tn"]],
    ]
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred +", "Pred -"])
    ax.set_yticks([0, 1], ["Gold +", "Gold -"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_fpr_fnr_by_split(splits: dict[str, Any], output_path: Path) -> None:
    plt = _import_matplotlib()
    names = list(splits)
    fpr = [float(splits[name]["classification_summary"]["fpr"]) for name in names]
    fnr = [float(splits[name]["classification_summary"]["fnr"]) for name in names]
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([index - 0.15 for index in x], fpr, width=0.3, label="FPR")
    ax.bar([index + 0.15 for index in x], fnr, width=0.3, label="FNR")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1)
    ax.set_title("FPR/FNR by Subgroup")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_margin_histograms(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    for bucket in ("TP", "TN", "FP", "FN"):
        margins = [float(row["margin"]) for row in rows if row["error_type"] == bucket]
        if margins:
            ax.hist(margins, bins=20, alpha=0.5, label=bucket)
    ax.set_title("Logit Margin by Error Bucket")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_probe_accuracy(probe_payload: dict[str, Any], output_path: Path) -> None:
    plt = _import_matplotlib()
    layers = sorted(int(layer) for layer in probe_payload["probe_results"]["layers"])
    overall = [probe_payload["probe_results"]["layers"][str(layer)]["all_examples"]["accuracy"] for layer in layers]
    fn_tp = [probe_payload["probe_results"]["layers"][str(layer)]["fn_vs_tp"]["accuracy"] for layer in layers]
    fp_tn = [probe_payload["probe_results"]["layers"][str(layer)]["fp_vs_tn"]["accuracy"] for layer in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, overall, label="All examples")
    ax.plot(layers, fn_tp, label="FN vs TP")
    ax.plot(layers, fp_tn, label="FP vs TN")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Layerwise Probe Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_patching_heatmap(aggregate: dict[str, Any], output_path: Path) -> None:
    plt = _import_matplotlib()
    matrix = aggregate.get("mean_normalized_effect", [])
    if not matrix:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Activation Patching Heatmap")
    ax.set_xlabel("Position")
    ax.set_ylabel("Layer")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ablation_head_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Render notebook-style layer/head ablation heatmap as interactive HTML."""

    head_rows = [row for row in rows if row.get("component_type") == "head"]
    if not head_rows:
        return

    max_layer = max(int(row["layer"]) for row in head_rows)
    max_head = max(int(row["head"]) for row in head_rows)
    values_by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in head_rows:
        values_by_cell[(int(row["layer"]), int(row["head"]))].append(float(row["delta_margin"]))

    matrix: list[list[float | None]] = []
    for layer in range(max_layer + 1):
        matrix.append(
            [
                mean(values_by_cell[(layer, head)]) if values_by_cell.get((layer, head)) else None
                for head in range(max_head + 1)
            ]
        )

    _write_notebook_imshow(
        matrix,
        output_path,
        x=list(range(max_head + 1)),
        y=list(range(max_layer + 1)),
        title="Mean Delta Margin After Head Ablation",
        xaxis="Head",
        yaxis="Layer",
    )


def plot_ablation_mlp_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Render notebook-style layer/error-bucket MLP ablation heatmap as interactive HTML."""

    mlp_rows = [row for row in rows if row.get("component_type") == "mlp"]
    if not mlp_rows:
        return

    bucket_order = ["TP", "TN", "FP", "FN"]
    max_layer = max(int(row["layer"]) for row in mlp_rows)
    values_by_cell: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in mlp_rows:
        values_by_cell[(int(row["layer"]), str(row["error_type"]))].append(float(row["delta_margin"]))

    matrix: list[list[float | None]] = []
    for layer in range(max_layer + 1):
        matrix.append(
            [
                mean(values_by_cell[(layer, bucket)]) if values_by_cell.get((layer, bucket)) else None
                for bucket in bucket_order
            ]
        )

    _write_notebook_imshow(
        matrix,
        output_path,
        x=bucket_order,
        y=list(range(max_layer + 1)),
        title="Mean Delta Margin After MLP Ablation",
        xaxis="Error Bucket",
        yaxis="Layer",
    )
