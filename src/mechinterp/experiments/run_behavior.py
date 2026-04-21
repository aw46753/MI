"""Behavior experiment entrypoint."""

from __future__ import annotations

from mechinterp.core.metrics import confusion_rates, summarize_logit_diffs
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, run_dir, write_csv, write_json
from mechinterp.tasks.ioi.analysis import build_corrupted_prompt
from mechinterp.tasks.ioi.score import score_prompt_with_candidates


def run(task_name: str, config_path: str) -> dict:
    """Run behavior scoring for a task."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    model = ModelWrapper(config)

    clean_rows_all: list[dict] = []
    corrupted_rows_all: list[dict] = []
    all_rows: list[dict] = []
    splits_payload: dict[str, dict] = {}

    for split in ("standard", "shifted"):
        dataset = task.build_dataset(split, config)
        clean_results = [task.score_example(model, example) for example in dataset]
        clean_rows = []
        corrupted_rows = []

        for example, clean_result in zip(dataset, clean_results, strict=True):
            clean_row = {
                **clean_result.to_flat_dict(),
                "pair_role": "clean",
                "expected_positive": True,
            }
            clean_rows.append(clean_row)
            clean_rows_all.append(clean_row)
            all_rows.append(clean_row)

            corrupted_prompt = build_corrupted_prompt(example)
            corrupted_result = score_prompt_with_candidates(
                model,
                corrupted_prompt,
                correct_token=example.correct_token,
                wrong_token=example.wrong_token,
                metadata={
                    **clean_result.metadata,
                    "prompt": corrupted_prompt,
                    "pair_role": "corrupted",
                },
            )
            corrupted_row = {
                **corrupted_result.to_flat_dict(),
                "pair_role": "corrupted",
                "expected_positive": False,
            }
            corrupted_rows.append(corrupted_row)
            corrupted_rows_all.append(corrupted_row)
            all_rows.append(corrupted_row)

        splits_payload[split] = {
            "clean_results": clean_rows,
            "corrupted_results": corrupted_rows,
            "clean_summary": summarize_logit_diffs(clean_rows),
            "corrupted_summary": summarize_logit_diffs(corrupted_rows),
            "classification_summary": confusion_rates(
                [float(row["logit_diff"]) for row in clean_rows],
                [float(row["logit_diff"]) for row in corrupted_rows],
            ),
        }

    behavior_dir = ensure_dir(run_dir(config, config_path) / "behavior")
    write_csv(behavior_dir / "results.csv", all_rows)
    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "results": clean_rows_all,
        "corrupted_results": corrupted_rows_all,
        "all_results": all_rows,
        "splits": splits_payload,
    }
    write_json(behavior_dir / "results.json", payload)
    return payload
