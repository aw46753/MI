from pathlib import Path

from mechinterp.core.runner import load_experiment_config


def test_runtime_device_override_wins_over_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name: gpt2",
                "device: cpu",
                "seed: 0",
                "dataset_sizes:",
                "  standard: 2",
                "  shifted: 2",
                "cache_hook_names:",
                "  - hook_resid_pre",
                "stop_at_layer: 2",
                "max_layer: 1",
                "output_dir: outputs",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(str(config_path), device="cuda:0")

    assert config.device == "cuda:0"
