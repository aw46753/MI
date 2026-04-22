from pathlib import Path
import re

import pytest
import torch

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
        value = int(text.strip())
        if value < 0 or value > 199:
            raise ValueError("not single token")
        return value

    def to_tokens(self, text, prepend_bos: bool = True):
        return torch.zeros((1, 16), dtype=torch.long)

    def __call__(self, tokens, return_type: str = "logits"):
        raise NotImplementedError("use forward_logits")

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, return_type="logits", **kwargs):
        logits = torch.zeros(1, tokens.shape[-1], 256)
        cache = {"blocks.0.hook_resid_pre": torch.ones(1, tokens.shape[-1], 3)}
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks=None, return_type="logits", **kwargs):
        return torch.zeros(1, tokens.shape[-1], 256)


class FakeTransformerBridge:
    @staticmethod
    def boot_transformers(model_name: str) -> FakeBridgeModel:
        assert model_name == "gpt2"
        return FakeBridgeModel()


def fake_forward_logits(self, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
    match = re.search(r"(\d+) \+ (\d+)", prompt)
    if match is None:
        raise ValueError(f"Could not parse addition prompt {prompt!r}")

    augend = int(match.group(1))
    addend = int(match.group(2))
    total = augend + addend

    logits = torch.zeros(1, 1, 256)
    logits[0, -1, total] = 3.0
    logits[0, -1, total - 1] = -2.0
    return logits


def test_addition_behavior_and_cache_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "addition.yaml"
    output_dir = tmp_path / "outputs"
    write_config(config_path, output_dir)

    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)
    monkeypatch.setattr("mechinterp.core.model.ModelWrapper.forward_logits", fake_forward_logits)

    behavior_payload = run_behavior("addition", str(config_path))
    assert behavior_payload["task"] == "addition"
    assert set(behavior_payload["splits"]) == {"standard", "shifted"}
    assert behavior_payload["results"]
    assert behavior_payload["negative_results"]
    assert behavior_payload["classification_summary"]["accuracy"] >= 0.0

    cache_metadata = run_cache("addition", str(config_path))
    assert cache_metadata["task"] == "addition"
    assert cache_metadata["selected_examples"]
    assert (output_dir / "addition" / "addition" / "cache" / "activations.pt").exists()
    assert (output_dir / "addition" / "addition" / "cache" / "metadata.json").exists()


def test_addition_patch_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "addition.yaml"
    output_dir = tmp_path / "outputs"
    write_config(config_path, output_dir)

    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)
    monkeypatch.setattr("mechinterp.core.model.ModelWrapper.forward_logits", fake_forward_logits)

    patch_payload = run_patch("addition", str(config_path))
    assert patch_payload["task"] == "addition"
    assert "aggregate" in patch_payload
