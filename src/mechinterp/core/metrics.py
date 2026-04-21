"""Shared metrics helpers."""

from __future__ import annotations

from statistics import mean
from typing import Any


def final_token_logit_diff(logits: Any, correct_token_id: int, wrong_token_id: int) -> float:
    """Compute final-token logit difference."""

    diff = logits[0, -1, correct_token_id] - logits[0, -1, wrong_token_id]
    return float(diff.item())


def prefers_correct(logit_diff: float) -> bool:
    """Return whether the model prefers the correct candidate."""
    return logit_diff > 0.0


def normalized_patch_effect(
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    patched_logit_diff: float,
) -> float:
    """Normalize patch effect into clean/corrupted recovery space."""

    denominator = clean_logit_diff - corrupted_logit_diff
    if denominator == 0:
        return 0.0
    return (patched_logit_diff - corrupted_logit_diff) / denominator


def summarize_logit_diffs(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize a collection of scored examples."""

    diffs = [float(row["logit_diff"]) for row in rows]
    if not diffs:
        return {"count": 0, "mean_logit_diff": 0.0, "preference_rate": 0.0}

    preference_rate = sum(1 for value in diffs if value > 0.0) / len(diffs)
    return {
        "count": len(diffs),
        "mean_logit_diff": mean(diffs),
        "preference_rate": preference_rate,
    }


def confusion_rates(
    positive_scores: list[float],
    negative_scores: list[float],
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Compute confusion-matrix counts and rates from logit differences.

    Positive scores are clean prompts that should be above the threshold.
    Negative scores are corrupted prompts that should stay at or below the threshold.
    """

    tp = sum(1 for score in positive_scores if score > threshold)
    fn = sum(1 for score in positive_scores if score <= threshold)
    fp = sum(1 for score in negative_scores if score > threshold)
    tn = sum(1 for score in negative_scores if score <= threshold)

    positive_count = len(positive_scores)
    negative_count = len(negative_scores)

    return {
        "threshold": threshold,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / positive_count if positive_count else 0.0,
        "fnr": fn / positive_count if positive_count else 0.0,
        "fpr": fp / negative_count if negative_count else 0.0,
        "tnr": tn / negative_count if negative_count else 0.0,
    }
