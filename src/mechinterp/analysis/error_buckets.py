"""Error bucket summaries and comparisons."""

from __future__ import annotations

from typing import Any

from mechinterp.evaluation.metrics import bucket_rows, bucket_summary, confusion_metrics, subgroup_metrics


def analyze_error_buckets(rows: list[dict[str, Any]], *, subgroup_key: str = "split") -> dict[str, Any]:
    """Return bucket summaries and subgroup confusion metrics."""

    return {
        "bucket_summary": bucket_summary(rows),
        "subgroup_metrics": subgroup_metrics(rows, subgroup_key),
        "confusion": confusion_metrics(rows),
        "bucket_counts": {bucket: len(bucket_rows(rows)[bucket]) for bucket in bucket_rows(rows)},
    }


def compare_bucket_pair(rows: list[dict[str, Any]], left_bucket: str, right_bucket: str) -> dict[str, float]:
    """Compare two error buckets by average margin and confidence."""

    buckets = bucket_rows(rows)
    left_rows = buckets.get(left_bucket, [])
    right_rows = buckets.get(right_bucket, [])

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    left_margin = _mean([float(row["margin"]) for row in left_rows])
    right_margin = _mean([float(row["margin"]) for row in right_rows])
    left_conf = _mean([float(row["confidence"]) for row in left_rows])
    right_conf = _mean([float(row["confidence"]) for row in right_rows])

    return {
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "left_mean_margin": left_margin,
        "right_mean_margin": right_margin,
        "margin_gap": left_margin - right_margin,
        "left_mean_confidence": left_conf,
        "right_mean_confidence": right_conf,
        "confidence_gap": left_conf - right_conf,
    }

