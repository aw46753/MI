import pytest
import torch

from mechinterp.tasks.ioi.score import score_prompt_with_candidates, validate_single_token_candidate


class FakeScoringModel:
    def __init__(self) -> None:
        self.token_map = {" Mary": 1, " John": 2}

    def to_single_token(self, text: str) -> int:
        if text not in self.token_map:
            raise ValueError("not single token")
        return self.token_map[text]

    def forward_logits(self, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
        logits = torch.zeros(1, 1, 5)
        logits[0, 0, self.token_map[" Mary"]] = 2.0
        logits[0, 0, self.token_map[" John"]] = -1.0
        return logits


def test_single_token_validation_accepts_and_rejects() -> None:
    model = FakeScoringModel()
    assert validate_single_token_candidate(model, " Mary") == 1
    with pytest.raises(ValueError):
        validate_single_token_candidate(model, " Mary Jane")


def test_scoring_output_shape_and_fields() -> None:
    model = FakeScoringModel()
    result = score_prompt_with_candidates(
        model,
        "When John and Mary went to the store, John gave the bag to",
        correct_token=" Mary",
        wrong_token=" John",
        metadata={"split": "standard", "template_id": "gave_object"},
    )

    assert result.prediction == " Mary"
    assert result.logit_diff == pytest.approx(3.0)
    assert result.correct_token_id == 1
    assert result.wrong_token_id == 2
    assert result.metadata["split"] == "standard"
