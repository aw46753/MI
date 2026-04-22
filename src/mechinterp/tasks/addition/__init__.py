"""Two-digit addition task package."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mechinterp.analysis.matched_pairs import build_matched_pairs_from_groups
from mechinterp.core.hooks import resid_pre_hook_name
from mechinterp.tasks.addition.data import AdditionExample, build_addition_dataset
from mechinterp.tasks.base import Task
from mechinterp.tasks.ioi.score import IOIScoreResult, score_prompt_with_candidates


class AdditionTask(Task):
    """Two-digit addition with no-carry and carry splits."""

    name = "addition"

    def split_names(self, config: Any) -> list[str]:
        return ["standard", "shifted"]

    def build_behavior_split(self, model: Any, split: str, config: Any) -> dict[str, Any]:
        dataset = self.build_dataset(split, config)
        clean_results = [self.score_example(model, example) for example in dataset]

        clean_rows = []
        corrupted_rows = []
        all_rows = []
        for example, clean_result in zip(dataset, clean_results, strict=True):
            clean_row = {
                **clean_result.to_flat_dict(),
                "pair_role": "clean",
                "expected_positive": True,
            }
            clean_rows.append(clean_row)
            all_rows.append(clean_row)

            corrupted_result = score_prompt_with_candidates(
                model,
                example.corrupted_prompt,
                correct_token=example.correct_token,
                wrong_token=example.wrong_token,
                metadata={
                    **clean_result.metadata,
                    "prompt": example.corrupted_prompt,
                    "pair_role": "corrupted",
                },
            )
            corrupted_row = {
                **corrupted_result.to_flat_dict(),
                "pair_role": "corrupted",
                "expected_positive": False,
            }
            corrupted_rows.append(corrupted_row)
            all_rows.append(corrupted_row)

        return {
            "primary_rows": clean_rows,
            "all_rows": all_rows,
            "positive_rows": clean_rows,
            "negative_rows": corrupted_rows,
            "positive_summary_name": "clean",
            "negative_summary_name": "corrupted",
        }

    def build_dataset(self, split: str, config: Any) -> list[AdditionExample]:
        return build_addition_dataset(split, config)

    def score_example(self, model: Any, example: AdditionExample) -> IOIScoreResult:
        return score_prompt_with_candidates(
            model,
            example.prompt,
            correct_token=example.correct_token,
            wrong_token=example.wrong_token,
            metadata={
                **asdict(example),
            },
        )

    def make_pairs(
        self,
        dataset: list[AdditionExample],
        scored_examples: list[IOIScoreResult],
        model: Any,
    ) -> list[Any]:
        return []

    def build_error_pairs(
        self,
        rows: list[dict[str, Any]],
        source_error_type: str,
        target_error_type: str,
        model: Any,
    ) -> list[Any]:
        source_rows = [row for row in rows if row["error_type"] == source_error_type]
        target_rows = [row for row in rows if row["error_type"] == target_error_type]
        return build_matched_pairs_from_groups(
            task_name=self.name,
            source_rows=source_rows,
            target_rows=target_rows,
            source_error_type=source_error_type,
            target_error_type=target_error_type,
            model=model,
            pair_score=lambda left, right: (
                0 if bool(left.get("carries")) != bool(right.get("carries")) else 1,
                abs(int(left.get("addend", 0)) - int(right.get("addend", 0))),
                abs(int(left.get("augend", 0)) - int(right.get("augend", 0))),
            ),
            metadata_keys=["augend", "addend", "total", "carries", "pair_role"],
            require_same_split=False,
        )

    def default_hook_names(self, config: Any) -> list[str]:
        return [resid_pre_hook_name(layer) for layer in range(config.patch.max_layer + 1)]
