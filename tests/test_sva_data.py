from mechinterp.core.config import CacheConfig, CompatibilityModeConfig, DatasetConfig, ExperimentConfig, OutputConfig, PatchConfig
from mechinterp.tasks.sva.analysis import build_corrupted_prompt
from mechinterp.tasks.sva.data import build_sva_dataset


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=17,
        dataset=DatasetConfig(dataset_sizes={"standard": 12, "shifted": 10}),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=4, cache_num_examples=2),
        patch=PatchConfig(max_layer=2),
        output=OutputConfig(output_dir="outputs"),
        compatibility_mode=CompatibilityModeConfig(),
    )


def test_sva_generation_is_deterministic() -> None:
    config = make_config()
    first = build_sva_dataset("standard", config)
    second = build_sva_dataset("standard", config)
    assert first == second


def test_sva_split_semantics_match_attractor_behavior() -> None:
    config = make_config()
    standard = build_sva_dataset("standard", config)
    shifted = build_sva_dataset("shifted", config)

    assert all(example.attractor is None for example in standard)
    assert all(example.attractor is not None for example in shifted)
    assert all(example.attractor_number != example.subject_number for example in shifted)


def test_sva_corruption_flips_governing_number() -> None:
    config = make_config()
    example = build_sva_dataset("shifted", config)[0]
    corrupted_prompt = build_corrupted_prompt(example)

    assert example.correct_token in {" is", " are"}
    assert example.subject in example.prompt
    assert corrupted_prompt != example.prompt


def test_sva_prompts_show_surface_variety() -> None:
    config = make_config()
    shifted = build_sva_dataset("shifted", config)

    assert len({example.context for example in shifted}) > 1
    assert len({example.intro for example in shifted}) > 1
