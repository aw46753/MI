"""Confusion-matrix and per-example evaluation utilities."""

from __future__ import annotations

from math import exp
from statistics import mean
from typing import Any


ERROR_BUCKETS = ("TP", "TN", "FP", "FN")


def probability_from_margin(margin: float) -> float:
    """Map a logit margin to a pseudo-probability for the positive class."""

    if margin >= 0:
        z = exp(-margin)
        return 1.0 / (1.0 + z)
    z = exp(margin)
    return z / (1.0 + z)


def error_type_from_labels(gold_label: int, predicted_label: int) -> str:
    """Return the confusion-matrix bucket for one prediction."""

    if gold_label == 1 and predicted_label == 1:
        return "TP"
    if gold_label == 0 and predicted_label == 0:
        return "TN"
    if gold_label == 0 and predicted_label == 1:
        return "FP"
    return "FN"


def annotate_prediction_rows(rows: list[dict[str, Any]], *, threshold: float = 0.0) -> list[dict[str, Any]]:
    """Annotate per-example rows with confusion-matrix metadata."""

    annotated: list[dict[str, Any]] = []
    for row in rows:
        margin = float(row["logit_diff"])
        gold_label = 1 if bool(row["expected_positive"]) else 0
        predicted_label = 1 if margin > threshold else 0
        positive_probability = probability_from_margin(margin)
        confidence = positive_probability if predicted_label == 1 else 1.0 - positive_probability

        annotated_row = dict(row)
        annotated_row.update(
            {
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "margin": margin,
                "confidence": confidence,
                "positive_probability": positive_probability,
                "error_type": error_type_from_labels(gold_label, predicted_label),
            }
        )
        annotated.append(annotated_row)
    return annotated


def bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split annotated rows into TP/TN/FP/FN buckets."""

    buckets = {bucket: [] for bucket in ERROR_BUCKETS}
    for row in rows:
        buckets[str(row["error_type"])].append(row)
    return buckets


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute classification metrics from annotated rows."""

    buckets = bucket_rows(rows)
    tp = len(buckets["TP"])
    tn = len(buckets["TN"])
    fp = len(buckets["FP"])
    fn = len(buckets["FN"])
    total = tp + tn + fp + fn

    return {
        "count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_ratio(tp + tn, total),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "tpr": _safe_ratio(tp, tp + fn),
        "tnr": _safe_ratio(tn, tn + fp),
        "fpr": _safe_ratio(fp, fp + tn),
        "fnr": _safe_ratio(fn, fn + tp),
    }


def bucket_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize confidence and margin statistics for each error bucket."""

    summaries: dict[str, dict[str, float]] = {}
    for bucket, bucket_rows_list in bucket_rows(rows).items():
        margins = [float(row["margin"]) for row in bucket_rows_list]
        confidences = [float(row["confidence"]) for row in bucket_rows_list]
        summaries[bucket] = {
            "count": len(bucket_rows_list),
            "mean_margin": mean(margins) if margins else 0.0,
            "mean_abs_margin": mean(abs(value) for value in margins) if margins else 0.0,
            "mean_confidence": mean(confidences) if confidences else 0.0,
        }
    return summaries


def subgroup_metrics(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    """Compute confusion metrics for each value of a metadata key."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_value = str(row.get(group_key, "unknown"))
        grouped.setdefault(group_value, []).append(row)

    return {group: confusion_metrics(group_rows) for group, group_rows in grouped.items()}
