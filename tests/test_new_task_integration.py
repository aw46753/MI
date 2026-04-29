from pathlib import Path
import re

import pytest
import torch

from mechinterp.core.runner import get_task
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.experiments.run_cache import run as run_cache
from mechinterp.experiments.run_patching import run as run_patch


def write_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "model_name: gpt2",
                "device: cpu",
                "seed: 3",
                "dataset_sizes:",
                "  standard: 8",
                "  shifted: 8",
                "cache_hook_names:",
                "  - hook_resid_pre",
                "stop_at_layer: 2",
                "max_layer: 1",
                "patch_position_mode: all",
                "cache_num_examples: 2",
                f"output_dir: {output_dir}",
            ]
        ),
        encoding="utf-8",
    )


class FakeBridgeModel:
    def __init__(self) -> None:
        self.compat_mode = None
        self.device = None

    def enable_compatibility_mode(self, **kwargs) -> None:
        self.compat_mode = kwargs

    def to(self, device: str, print_details: bool = False) -> None:
        self.device = device

    def eval(self) -> None:
        return None

    def to_single_token(self, text: str) -> int:
        mapping = {" yes": 1, " no": 2, " is": 3, " are": 4}
        if text not in mapping:
            raise ValueError("not single token")
        return mapping[text]

    def to_tokens(self, text, prepend_bos: bool = True):
        return torch.zeros((1, 20), dtype=torch.long)

    def to_str_tokens(self, text, prepend_bos: bool = True):
        return ["tok"] * 20

    def __call__(self, tokens, return_type: str = "logits"):
        raise NotImplementedError("use forward_logits")

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, return_type="logits", **kwargs):
        logits = torch.zeros(1, tokens.shape[-1], 8)
        cache = {"blocks.0.hook_resid_pre": torch.ones(1, tokens.shape[-1], 3)}
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks=None, return_type="logits", **kwargs):
        return torch.zeros(1, tokens.shape[-1], 8)


class FakeTransformerBridge:
    @staticmethod
    def boot_transformers(model_name: str) -> FakeBridgeModel:
        assert model_name == "gpt2"
        return FakeBridgeModel()


def fake_forward_logits(self, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
    logits = torch.zeros(1, 1, 8)
    prompt_lower = prompt.lower()
    if "greater than" in prompt_lower:
        numbers = [int(match) for match in re.findall(r"\d+", prompt_lower)]
        left, right = numbers[0], numbers[1]
        yes_token, no_token = 1, 2
        if left > right:
            logits[0, -1, yes_token] = 3.0
            logits[0, -1, no_token] = -2.0
        else:
            logits[0, -1, yes_token] = -2.0
            logits[0, -1, no_token] = 3.0
        return logits

    is_token, are_token = 3, 4
    if "children" in prompt_lower or "dogs" in prompt_lower or "cats" in prompt_lower or "birds" in prompt_lower:
        logits[0, -1, is_token] = -2.0
        logits[0, -1, are_token] = 3.0
    else:
        logits[0, -1, is_token] = 3.0
        logits[0, -1, are_token] = -2.0
    return logits


def test_new_tasks_are_registered() -> None:
    assert get_task("greater_than").name == "greater_than"
    assert get_task("sva").name == "sva"


@pytest.mark.parametrize("task_name", ["greater_than", "sva"])
def test_new_task_behavior_cache_and_patch_outputs(
    task_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / f"{task_name}.yaml"
    output_dir = tmp_path / "outputs"
    write_config(config_path, output_dir)

    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)
    monkeypatch.setattr("mechinterp.core.model.ModelWrapper.forward_logits", fake_forward_logits)

    behavior_payload = run_behavior(task_name, str(config_path))
    assert behavior_payload["task"] == task_name
    assert set(behavior_payload["splits"]) == {"standard", "shifted"}
    assert behavior_payload["results"]
    assert behavior_payload["negative_results"]

    cache_metadata = run_cache(task_name, str(config_path))
    assert cache_metadata["task"] == task_name
    assert cache_metadata["selected_examples"]

    patch_payload = run_patch(task_name, str(config_path))
    assert patch_payload["task"] == task_name
    assert "aggregate" in patch_payload
