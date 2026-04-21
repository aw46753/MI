"""IOI task package."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mechinterp.tasks.base import Task
from mechinterp.tasks.ioi.analysis import build_matched_pairs, default_residual_hook_targets
from mechinterp.tasks.ioi.data import IOIExample, build_ioi_dataset
from mechinterp.tasks.ioi.score import IOIScoreResult, score_prompt_with_candidates


class IOITask(Task):
    """Indirect Object Identification task."""

    name = "ioi"

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

    def default_hook_names(self, config: Any) -> list[str]:
        return default_residual_hook_targets(config.patch.max_layer)
