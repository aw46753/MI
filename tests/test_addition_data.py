from mechinterp.core.config import CacheConfig, CompatibilityModeConfig, DatasetConfig, ExperimentConfig, OutputConfig, PatchConfig
from mechinterp.tasks.addition.data import build_addition_dataset


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=11,
        dataset=DatasetConfig(dataset_sizes={"standard": 12, "shifted": 10}),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=2, cache_num_examples=2),
        patch=PatchConfig(max_layer=0),
        output=OutputConfig(output_dir="outputs"),
        compatibility_mode=CompatibilityModeConfig(),
    )


def test_addition_generation_is_deterministic() -> None:
    config = make_config()
    first = build_addition_dataset("standard", config)
    second = build_addition_dataset("standard", config)
    assert first == second


def test_addition_split_semantics_match_carry_behavior() -> None:
    config = make_config()
    standard = build_addition_dataset("standard", config)
    shifted = build_addition_dataset("shifted", config)

    assert all(not example.carries for example in standard)
    assert all(example.carries for example in shifted)
    assert all((example.augend % 10) + (example.addend % 10) < 10 for example in standard)
    assert all((example.augend % 10) + (example.addend % 10) >= 10 for example in shifted)


def test_addition_examples_include_off_by_one_corruption() -> None:
    config = make_config()
    example = build_addition_dataset("standard", config)[0]

    assert example.correct_token.startswith(" ")
    assert example.wrong_token == f" {example.total - 1}"
    assert f"{example.augend} + {example.addend}" in example.prompt
    assert f"{example.augend - 1} + {example.addend}" in example.corrupted_prompt
