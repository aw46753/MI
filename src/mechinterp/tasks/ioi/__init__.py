"""IOI task package."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mechinterp.analysis.matched_pairs import build_matched_pairs_from_groups
from mechinterp.tasks.base import Task
from mechinterp.tasks.ioi.analysis import build_corrupted_prompt, build_matched_pairs, default_residual_hook_targets
from mechinterp.tasks.ioi.data import IOIExample, build_ioi_dataset
from mechinterp.tasks.ioi.score import (
    IOIScoreResult,
    find_invalid_single_token_candidates,
    score_prompt_with_candidates,
)


class IOITask(Task):
    """Indirect Object Identification task."""

    name = "ioi"

    def split_names(self, config: Any) -> list[str]:
        return ["standard", "shifted"]

    def build_behavior_split(self, model: Any, split: str, config: Any) -> dict[str, Any]:
        invalid_candidates = find_invalid_single_token_candidates(
            model,
            [f" {name}" for name in config.dataset.names],
        )
        if invalid_candidates:
            invalid_names = ", ".join(repr(candidate.strip()) for candidate in invalid_candidates)
            raise ValueError(
                "IOI config contains names that are not single GPT-2 tokens: "
                f"{invalid_names}. Update `names` in the config to use token-safe names."
            )

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
            all_rows.append(corrupted_row)

        return {
            "primary_rows": clean_rows,
            "all_rows": all_rows,
            "positive_rows": clean_rows,
            "negative_rows": corrupted_rows,
            "positive_summary_name": "clean",
            "negative_summary_name": "corrupted",
        }

    def build_dataset(self, split: str, config: Any) -> list[IOIExample]:
        return build_ioi_dataset(split, config)

    def score_example(self, model: Any, example: IOIExample) -> IOIScoreResult:
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
        dataset: list[IOIExample],
        scored_examples: list[IOIScoreResult],
        model: Any,
    ) -> list[Any]:
        return build_matched_pairs(dataset, scored_examples, model)

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
                0 if left.get("template_id") == right.get("template_id") else 10,
                abs(len(str(left["prompt"])) - len(str(right["prompt"]))),
            ),
            metadata_keys=["template_id", "subject", "indirect_object", "pair_role"],
        )

    def default_hook_names(self, config: Any) -> list[str]:
        return default_residual_hook_targets(config.patch.max_layer)
