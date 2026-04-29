import pytest
import torch
from dataclasses import replace

from mechinterp.core.config import (
    CacheConfig,
    CompatibilityModeConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PatchConfig,
)
from mechinterp.core.model import ModelWrapper, resolve_device


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

    def to_tokens(self, text, prepend_bos: bool = True):
        return torch.tensor([[0, 1, 2]])

    def to_str_tokens(self, text, prepend_bos: bool = True):
        return ["<BOS>", "When", "Mary"]

    def to_single_token(self, text: str) -> int:
        return 1

    def __call__(self, tokens, return_type: str = "logits"):
        logits = torch.zeros(1, 3, 4)
        logits[0, -1, 1] = 1.0
        return logits

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, return_type="logits", **kwargs):
        logits = self(tokens, return_type=return_type)
        cache = {"blocks.0.hook_resid_pre": torch.ones(1, 3, 2)}
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks=None, return_type="logits", **kwargs):
        return self(tokens, return_type=return_type)


class FakeTransformerBridge:
    @staticmethod
    def boot_transformers(model_name: str) -> FakeBridgeModel:
        assert model_name == "gpt2"
        return FakeBridgeModel()


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=0,
        dataset=DatasetConfig(
            dataset_sizes={"standard": 2, "shifted": 2},
            names=["Alice", "Mary", "John", "James"],
            templates=["gave_object", "handed_object"],
            shifted_name_count=2,
            shifted_template_count=1,
        ),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=2, cache_num_examples=1),
        patch=PatchConfig(max_layer=2),
        output=OutputConfig(output_dir="outputs"),
        compatibility_mode=CompatibilityModeConfig(),
    )


def test_model_wrapper_boot_and_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)

    wrapper = ModelWrapper(make_config())
    model = wrapper.load()
    tokens = wrapper.to_tokens("hello")
    str_tokens = wrapper.to_str_tokens("hello")
    logits = wrapper.forward_logits("hello")
    _, cache = wrapper.run_with_cache("hello", names_filter=lambda name: True, stop_at_layer=1)
    hooked_logits = wrapper.run_with_hooks("hello", fwd_hooks=[])

    assert model.device == "cpu"
    assert tokens.shape == (1, 3)
    assert str_tokens[-1] == "Mary"
    assert logits.shape == (1, 3, 4)
    assert "blocks.0.hook_resid_pre" in cache
    assert hooked_logits.shape == (1, 3, 4)


def test_model_wrapper_uses_explicit_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)
    monkeypatch.setattr("mechinterp.core.model.resolve_device", lambda device: device)

    config = replace(make_config(), device="cuda:0")
    wrapper = ModelWrapper(config)

    model = wrapper.load()

    assert model.device == "cuda:0"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device requested but CUDA is not available"):
        resolve_device("cuda")
