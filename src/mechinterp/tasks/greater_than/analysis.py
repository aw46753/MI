"""Analysis helpers for greater-than patching."""

from __future__ import annotations

from typing import Any

from mechinterp.core.hooks import resid_pre_hook_name
from mechinterp.core.pairs import MatchedPair, token_lengths_match
from mechinterp.tasks.greater_than.data import GreaterThanExample, render_greater_than_prompt
from mechinterp.tasks.ioi.score import IOIScoreResult, score_prompt_with_candidates


def build_corrupted_prompt(example: GreaterThanExample) -> str:
    """Return the precomputed corrupted prompt."""

    return render_greater_than_prompt(example.corrupted_left, example.corrupted_right)


def default_residual_hook_targets(max_layer: int) -> list[str]:
    """Return resid_pre hooks for a simple patch sweep."""

    return [resid_pre_hook_name(layer) for layer in range(max_layer + 1)]


def build_matched_pairs(
    dataset: list[GreaterThanExample],
    scored_examples: list[IOIScoreResult],
    model: Any,
) -> list[MatchedPair]:
    """Build matched clean/corrupted pairs for patching."""

    pairs: list[MatchedPair] = []
    for index, (example, clean_score) in enumerate(zip(dataset, scored_examples, strict=True)):
        corrupted_prompt = build_corrupted_prompt(example)
        if not token_lengths_match(model, example.prompt, corrupted_prompt):
            continue

        corrupted_score = score_prompt_with_candidates(
            model,
            corrupted_prompt,
            correct_token=example.correct_token,
            wrong_token=example.wrong_token,
            metadata={
                "prompt": corrupted_prompt,
                "split": example.split,
                "left": example.corrupted_left,
                "right": example.corrupted_right,
                "gap": abs(example.corrupted_left - example.corrupted_right),
                "pair_role": "corrupted",
            },
        )

        if clean_score.logit_diff <= 0.0 or corrupted_score.logit_diff >= 0.0:
            continue

        pairs.append(
            MatchedPair(
                pair_id=f"{example.split}-{index}",
                task_name="greater_than",
                split=example.split,
                template_id="greater_than",
                clean_prompt=example.prompt,
                corrupted_prompt=corrupted_prompt,
                correct_token=example.correct_token,
                wrong_token=example.wrong_token,
                clean_logit_diff=clean_score.logit_diff,
                corrupted_logit_diff=corrupted_score.logit_diff,
                metadata={
                    "left": example.left,
                    "right": example.right,
                    "gap": example.gap,
                },
            )
        )

    return pairs
