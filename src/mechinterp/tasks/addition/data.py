"""Synthetic two-digit addition dataset generation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from mechinterp.core.config import ExperimentConfig


@dataclass(frozen=True)
class AdditionExample:
    """One synthetic two-digit addition prompt."""

    prompt: str
    corrupted_prompt: str
    correct_token: str
    wrong_token: str
    augend: int
    addend: int
    total: int
    split: str
    carries: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def render_addition_prompt(augend: int, addend: int) -> str:
    """Render a plain addition prompt."""

    return f"Question: What is {augend} + {addend}?\nAnswer:"


def _is_valid_candidate(augend: int, addend: int, *, carries: bool) -> bool:
    total = augend + addend
    if augend < 11 or addend < 10:
        return False
    if total < 20 or total > 99:
        return False
    if ((augend % 10) + (addend % 10) >= 10) != carries:
        return False
    return True


def _candidate_pool(carries: bool) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for augend in range(11, 90):
        for addend in range(10, 90):
            if _is_valid_candidate(augend, addend, carries=carries):
                pairs.append((augend, addend))
    return pairs


def build_addition_dataset(split: str, config: ExperimentConfig) -> list[AdditionExample]:
    """Build a deterministic addition dataset for one split."""

    if split not in {"standard", "shifted"}:
        raise ValueError(f"Unknown split '{split}'")

    carries = split == "shifted"
    candidate_pairs = _candidate_pool(carries)
    size = config.dataset.dataset_sizes[split]
    if size > len(candidate_pairs):
        raise ValueError(
            f"Requested {size} examples for split '{split}', but only {len(candidate_pairs)} are available"
        )

    rng = random.Random(f"{config.seed}:{split}:addition")
    rng.shuffle(candidate_pairs)

    examples: list[AdditionExample] = []
    for augend, addend in candidate_pairs[:size]:
        total = augend + addend
        examples.append(
            AdditionExample(
                prompt=render_addition_prompt(augend, addend),
                corrupted_prompt=render_addition_prompt(augend - 1, addend),
                correct_token=f" {total}",
                wrong_token=f" {total - 1}",
                augend=augend,
                addend=addend,
                total=total,
                split=split,
                carries=carries,
            )
        )

    return examples
