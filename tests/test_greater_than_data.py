from mechinterp.core.config import CacheConfig, CompatibilityModeConfig, DatasetConfig, ExperimentConfig, OutputConfig, PatchConfig
from mechinterp.tasks.greater_than.analysis import build_corrupted_prompt
from mechinterp.tasks.greater_than.data import build_greater_than_dataset


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=13,
        dataset=DatasetConfig(dataset_sizes={"standard": 12, "shifted": 10}),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=4, cache_num_examples=2),
        patch=PatchConfig(max_layer=2),
        output=OutputConfig(output_dir="outputs"),
        compatibility_mode=CompatibilityModeConfig(),
    )


def test_greater_than_generation_is_deterministic() -> None:
    config = make_config()
    first = build_greater_than_dataset("standard", config)
    second = build_greater_than_dataset("standard", config)
    assert first == second


def test_greater_than_split_semantics_match_gap_regime() -> None:
    config = make_config()
    standard = build_greater_than_dataset("standard", config)
    shifted = build_greater_than_dataset("shifted", config)

    assert min(example.gap for example in standard) >= 12
    assert max(example.gap for example in shifted) <= 4


def test_greater_than_corruption_flips_label_and_preserves_prompt_shape() -> None:
    config = make_config()
    example = build_greater_than_dataset("standard", config)[0]
    corrupted_prompt = build_corrupted_prompt(example)

    assert example.correct_token in {" yes", " no"}
    assert example.answer_is_yes != example.corrupted_answer_is_yes
    assert str(example.left) in example.prompt
    assert str(example.corrupted_left) in corrupted_prompt


def test_greater_than_prompts_show_template_variety() -> None:
    config = make_config()
    examples = build_greater_than_dataset("standard", config)

    assert len({example.template_id for example in examples}) > 1
