"""Scoring for IOI prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mechinterp.core.metrics import final_token_logit_diff


@dataclass(frozen=True)
class IOIScoreResult:
    """Structured score output for an IOI prompt."""

    prediction: str
    logit_diff: float
    correct_token_id: int
    wrong_token_id: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly representation."""
        row = {
            "prediction": self.prediction,
            "logit_diff": self.logit_diff,
            "correct_token_id": self.correct_token_id,
            "wrong_token_id": self.wrong_token_id,
        }
        for key, value in self.metadata.items():
            row[key] = value
        return row


def validate_single_token_candidate(model: Any, candidate: str) -> int:
    """Validate that a candidate string maps to a single GPT-2 token."""

    try:
        return int(model.to_single_token(candidate))
    except Exception as exc:
        raise ValueError(
            f"Candidate token {candidate!r} is not a single GPT-2 token. "
            "Use leading-space names that tokenize to one token."
        ) from exc


def score_prompt_with_candidates(
    model: Any,
    prompt: str,
    *,
    correct_token: str,
    wrong_token: str,
    metadata: dict[str, Any],
) -> IOIScoreResult:
    """Score a prompt using fixed candidate tokens."""

    correct_token_id = validate_single_token_candidate(model, correct_token)
    wrong_token_id = validate_single_token_candidate(model, wrong_token)

    logits = model.forward_logits(prompt, prepend_bos=True)
    logit_diff = final_token_logit_diff(logits, correct_token_id, wrong_token_id)
    prediction = correct_token if logit_diff >= 0.0 else wrong_token

    return IOIScoreResult(
        prediction=prediction,
        logit_diff=logit_diff,
        correct_token_id=correct_token_id,
        wrong_token_id=wrong_token_id,
        metadata=metadata,
    )
