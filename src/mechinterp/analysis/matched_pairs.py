"""Matched-pair helpers for error-bucket patching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mechinterp.core.pairs import token_lengths_match


@dataclass
class ErrorMatchedPair:
    """A matched source/target pair for error-focused patching."""

    pair_id: str
    task_name: str
    split: str
    source_error_type: str
    target_error_type: str
    source_prompt: str
    target_prompt: str
    correct_token: str
    wrong_token: str
    source_logit_diff: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_matched_pairs_from_groups(
    *,
    task_name: str,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_error_type: str,
    target_error_type: str,
    model: Any,
    pair_score: Any,
    metadata_keys: list[str] | None = None,
    require_same_split: bool = True,
) -> list[ErrorMatchedPair]:
    """Greedily build matched pairs using a task-provided scoring function."""

    metadata_keys = metadata_keys or []
    pairs: list[ErrorMatchedPair] = []
    for index, source_row in enumerate(source_rows):
        candidates = []
        for target_row in target_rows:
            if require_same_split and source_row["split"] != target_row["split"]:
                continue
            if source_row["pair_role"] != target_row["pair_role"]:
                continue
            if not token_lengths_match(model, source_row["prompt"], target_row["prompt"]):
                continue
            candidates.append(target_row)

        if not candidates:
            continue

        target_row = min(candidates, key=lambda row: pair_score(source_row, row))
        metadata = {key: source_row.get(key) for key in metadata_keys}
        pairs.append(
            ErrorMatchedPair(
                pair_id=f"{source_error_type}-{target_error_type}-{index}",
                task_name=task_name,
                split=str(source_row["split"]),
                source_error_type=source_error_type,
                target_error_type=target_error_type,
                source_prompt=str(source_row["prompt"]),
                target_prompt=str(target_row["prompt"]),
                correct_token=str(source_row["correct_token"]),
                wrong_token=str(source_row["wrong_token"]),
                source_logit_diff=float(source_row["margin"]),
                metadata=metadata,
            )
        )
    return pairs
