from mechinterp.evaluation.metrics import annotate_prediction_rows, bucket_summary, confusion_metrics


def test_annotation_and_confusion_metrics_cover_all_error_types() -> None:
    rows = annotate_prediction_rows(
        [
            {"logit_diff": 2.0, "expected_positive": True},
            {"logit_diff": -1.0, "expected_positive": True},
            {"logit_diff": 3.0, "expected_positive": False},
            {"logit_diff": -2.0, "expected_positive": False},
        ]
    )

    summary = confusion_metrics(rows)
    buckets = bucket_summary(rows)

    assert [row["error_type"] for row in rows] == ["TP", "FN", "FP", "TN"]
    assert summary["tp"] == 1
    assert summary["fn"] == 1
    assert summary["fp"] == 1
    assert summary["tn"] == 1
    assert summary["accuracy"] == 0.5
    assert buckets["TP"]["count"] == 1
    assert buckets["FN"]["count"] == 1
