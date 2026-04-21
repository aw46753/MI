"""Helpers for matched clean/corrupted pairs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MatchedPair:
    """Matched clean/corrupted prompts for patching."""

    pair_id: str
    task_name: str
    split: str
    template_id: str
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    wrong_token: str
    clean_logit_diff: float
    corrupted_logit_diff: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def token_lengths_match(model: Any, clean_prompt: str, corrupted_prompt: str) -> bool:
    """Return whether two prompts tokenize to the same length."""

    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)
    return int(clean_tokens.shape[-1]) == int(corrupted_tokens.shape[-1])
