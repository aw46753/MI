"""Synthetic greater-than dataset generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from mechinterp.core.config import ExperimentConfig


POSITIVE_TOKEN = " yes"
NEGATIVE_TOKEN = " no"
GREATER_THAN_PROMPT_TEMPLATES = (
    "Question: Is {left} greater than {right}?\nAnswer:",
    "Decide: is {left} greater than {right}?\nAnswer:",
    "Comparison check: is {left} greater than {right}?\nAnswer:",
    "True or false: {left} is greater than {right}.\nAnswer:",
)


@dataclass(frozen=True)
class GreaterThanExample:
    """One synthetic numeric comparison prompt."""

    prompt: str
    corrupted_prompt: str
    correct_token: str
    wrong_token: str
    left: int
    right: int
    corrupted_left: int
    corrupted_right: int
    answer_is_yes: bool
    corrupted_answer_is_yes: bool
    gap: int
    split: str
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def render_greater_than_prompt(left: int, right: int, *, template_id: int = 0) -> str:
    """Render a binary greater-than comparison prompt."""

    return GREATER_THAN_PROMPT_TEMPLATES[template_id].format(left=left, right=right)


def _candidate_pairs(*, small_gap: bool) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(20, 100):
        for right in range(20, 100):
            if left == right:
                continue
            gap = abs(left - right)
            if small_gap and 1 <= gap <= 4:
                pairs.append((left, right))
            elif not small_gap and gap >= 12:
                pairs.append((left, right))
    return pairs


def _find_corruption(left: int, right: int) -> tuple[int, int]:
    gap = abs(left - right)
    if left > right:
        corrupted_left = max(10, right - gap)
        if len(str(corrupted_left)) == len(str(left)) and corrupted_left < right:
            return corrupted_left, right
        corrupted_right = min(99, left + gap)
        if len(str(corrupted_right)) == len(str(right)) and left < corrupted_right:
            return left, corrupted_right
    else:
        corrupted_left = min(99, right + gap)
        if len(str(corrupted_left)) == len(str(left)) and corrupted_left > right:
            return corrupted_left, right
        corrupted_right = max(10, left - gap)
        if len(str(corrupted_right)) == len(str(right)) and left > corrupted_right:
            return left, corrupted_right
    raise ValueError(f"Could not build stable corruption for pair ({left}, {right})")


def build_greater_than_dataset(split: str, config: ExperimentConfig) -> list[GreaterThanExample]:
    """Build a deterministic greater-than dataset for one split."""

    if split not in {"standard", "shifted"}:
        raise ValueError(f"Unknown split '{split}'")

    small_gap = split == "shifted"
    candidate_pairs = _candidate_pairs(small_gap=small_gap)
    size = config.dataset.dataset_sizes[split]
    if size > len(candidate_pairs):
        raise ValueError(
            f"Requested {size} examples for split '{split}', but only {len(candidate_pairs)} are available"
        )

    rng = random.Random(f"{config.seed}:{split}:greater_than")
    rng.shuffle(candidate_pairs)

    examples: list[GreaterThanExample] = []
    for left, right in candidate_pairs:
        corrupted_left, corrupted_right = _find_corruption(left, right)
        answer_is_yes = left > right
        corrupted_answer_is_yes = corrupted_left > corrupted_right
        if answer_is_yes == corrupted_answer_is_yes:
            continue
        template_id = rng.randrange(len(GREATER_THAN_PROMPT_TEMPLATES))
        examples.append(
            GreaterThanExample(
                prompt=render_greater_than_prompt(left, right, template_id=template_id),
                corrupted_prompt=render_greater_than_prompt(corrupted_left, corrupted_right, template_id=template_id),
                correct_token=POSITIVE_TOKEN if answer_is_yes else NEGATIVE_TOKEN,
                wrong_token=NEGATIVE_TOKEN if answer_is_yes else POSITIVE_TOKEN,
                left=left,
                right=right,
                corrupted_left=corrupted_left,
                corrupted_right=corrupted_right,
                answer_is_yes=answer_is_yes,
                corrupted_answer_is_yes=corrupted_answer_is_yes,
                gap=abs(left - right),
                split=split,
                template_id=f"template_{template_id}",
            )
        )
        if len(examples) >= size:
            break

    if len(examples) < size:
        raise ValueError(f"Could only build {len(examples)} examples for split '{split}', requested {size}")

    return examples
