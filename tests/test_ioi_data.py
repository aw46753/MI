from mechinterp.core.config import (
    CacheConfig,
    CompatibilityModeConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PatchConfig,
)
from mechinterp.tasks.ioi.data import build_ioi_dataset


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=7,
        dataset=DatasetConfig(
            dataset_sizes={"standard": 6, "shifted": 4},
            names=["Alice", "Mary", "John", "James", "Laura", "Sarah"],
            templates=["gave_object", "handed_object", "brought_object"],
            shifted_name_count=2,
            shifted_template_count=1,
        ),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=4, cache_num_examples=2),
        patch=PatchConfig(max_layer=4),
        output=OutputConfig(output_dir="outputs"),
        compatibility_mode=CompatibilityModeConfig(),
    )


def test_ioi_generation_is_deterministic() -> None:
    config = make_config()
    first = build_ioi_dataset("standard", config)
    second = build_ioi_dataset("standard", config)
    assert first == second


def test_shifted_split_uses_held_out_resources() -> None:
    config = make_config()
    standard = build_ioi_dataset("standard", config)
    shifted = build_ioi_dataset("shifted", config)

    standard_names = {example.subject for example in standard} | {
        example.indirect_object for example in standard
    }
    shifted_names = {example.subject for example in shifted} | {
        example.indirect_object for example in shifted
    }
    standard_templates = {example.template_id for example in standard}
    shifted_templates = {example.template_id for example in shifted}

    assert standard_names.isdisjoint(shifted_names)
    assert standard_templates.isdisjoint(shifted_templates)


def test_ioi_example_metadata_fields_are_present() -> None:
    config = make_config()
    example = build_ioi_dataset("standard", config)[0]
    assert example.prompt
    assert example.correct_token.startswith(" ")
    assert example.wrong_token.startswith(" ")
    assert example.template_id in config.dataset.templates
    assert example.split == "standard"
