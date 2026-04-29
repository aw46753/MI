"""Analysis helpers for SVA patching."""

from __future__ import annotations

from typing import Any

from mechinterp.core.hooks import resid_pre_hook_name
from mechinterp.core.pairs import MatchedPair, token_lengths_match
from mechinterp.tasks.ioi.score import IOIScoreResult, score_prompt_with_candidates
from mechinterp.tasks.sva.data import SVAExample


def build_corrupted_prompt(example: SVAExample) -> str:
    """Return the precomputed corrupted prompt."""

    return example.corrupted_prompt


def default_residual_hook_targets(max_layer: int) -> list[str]:
    """Return resid_pre hooks for a simple patch sweep."""

    return [resid_pre_hook_name(layer) for layer in range(max_layer + 1)]


def build_matched_pairs(
    dataset: list[SVAExample],
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
                "subject_number": "plural" if example.subject_number == "singular" else "singular",
                "context": example.context,
                "intro": example.intro,
                "attractor_link": example.attractor_link,
                "pair_role": "corrupted",
            },
        )

        if clean_score.logit_diff <= 0.0 or corrupted_score.logit_diff >= 0.0:
            continue

        pairs.append(
            MatchedPair(
                pair_id=f"{example.split}-{index}",
                task_name="sva",
                split=example.split,
                template_id="sva",
                clean_prompt=example.prompt,
                corrupted_prompt=corrupted_prompt,
                correct_token=example.correct_token,
                wrong_token=example.wrong_token,
                clean_logit_diff=clean_score.logit_diff,
                corrupted_logit_diff=corrupted_score.logit_diff,
                metadata={
                    "subject": example.subject,
                    "subject_number": example.subject_number,
                    "attractor": example.attractor,
                    "attractor_number": example.attractor_number,
                },
            )
        )

    return pairs
