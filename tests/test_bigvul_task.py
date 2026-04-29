import json
from pathlib import Path

import pytest
import torch

from mechinterp.core.config import (
    CacheConfig,
    CompatibilityModeConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PatchConfig,
)
from mechinterp.core.runner import get_task, read_json
from mechinterp.experiments.run_ablation import run as run_ablation
from mechinterp.experiments.run_cache import run as run_cache
from mechinterp.experiments.run_error_analysis import run as run_analyze
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.experiments.run_patching import run as run_patch
from mechinterp.experiments.run_probes import run as run_probe
from mechinterp.tasks.bigvul import BigVulTask
from mechinterp.tasks.bigvul.data import build_bigvul_dataset, preprocess_bigvul_dataset


def make_config(dataset_path: Path, output_dir: Path, *, pairs_per_cwe: int = 100, seed: int = 11) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gpt2",
        device="cpu",
        seed=seed,
        dataset=DatasetConfig(
            dataset_path=str(dataset_path),
            target_cwes=["CWE-119", "CWE-20"],
            pairs_per_cwe=pairs_per_cwe,
            split_mode="by_cwe",
        ),
        cache=CacheConfig(cache_hook_names=["hook_resid_pre"], stop_at_layer=2, cache_num_examples=2),
        patch=PatchConfig(max_layer=1),
        output=OutputConfig(output_dir=str(output_dir)),
        compatibility_mode=CompatibilityModeConfig(),
    )


class FakeBridgeModel:
    def __init__(self) -> None:
        self.device = None
        self.compat_mode = None
        self.cfg = type("Cfg", (), {"n_layers": 2, "n_heads": 2})()

    def enable_compatibility_mode(self, **kwargs) -> None:
        self.compat_mode = kwargs

    def to(self, device: str, print_details: bool = False) -> None:
        self.device = device

    def eval(self) -> None:
        return None

    def to_single_token(self, text: str) -> int:
        mapping = {" yes": 1, " no": 2}
        if text not in mapping:
            raise ValueError(f"unsupported token {text}")
        return mapping[text]

    def to_tokens(self, text, prepend_bos: bool = True):
        return torch.zeros((1, 16), dtype=torch.long)

    def to_str_tokens(self, text, prepend_bos: bool = True):
        return ["tok"] * 16

    def __call__(self, tokens, return_type: str = "logits"):
        return torch.zeros(1, tokens.shape[-1], 8)

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, return_type="logits", **kwargs):
        logits = self(tokens, return_type=return_type)
        cache = {
            "blocks.0.hook_in": torch.ones(1, tokens.shape[-1], 4),
            "blocks.1.hook_in": torch.full((1, tokens.shape[-1], 4), 2.0),
        }
        if names_filter is not None:
            cache = {key: value for key, value in cache.items() if names_filter(key)}
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks=None, return_type="logits", **kwargs):
        logits = torch.zeros(1, tokens.shape[-1], 8)
        logits[0, -1, 1] = 0.5
        logits[0, -1, 2] = -0.5
        return logits


class FakeTransformerBridge:
    @staticmethod
    def boot_transformers(model_name: str) -> FakeBridgeModel:
        assert model_name == "gpt2"
        return FakeBridgeModel()


def fake_forward_logits(self, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
    logits = torch.zeros(1, 1, 8)
    if "vulnerable_" in prompt:
        logits[0, -1, 1] = 3.0
        logits[0, -1, 2] = -2.0
    else:
        logits[0, -1, 1] = -2.0
        logits[0, -1, 2] = 3.0
    return logits


def write_bigvul_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def build_raw_rows(*, per_cwe: int) -> list[dict]:
    rows: list[dict] = []
    for cwe_id in ("119", "20"):
        for index in range(per_cwe):
            rows.append(
                {
                    "id": f"{cwe_id}-{index}",
                    "commit_id": f"commit-{cwe_id}-{index}",
                    "cwe_id": cwe_id,
                    "func_before": (
                        f"int vulnerable_{cwe_id}_{index}(char *buf) {{\n"
                        f"  return {index};\n"
                        "}\n"
                    ),
                    "func_after": (
                        f"int patched_{cwe_id}_{index}(char *buf) {{\n"
                        f"  return {index + 1};\n"
                        "}\n"
                    ),
                    "project": f"project-{cwe_id}",
                    "file_path": f"src/{cwe_id}/{index}.c",
                    "function_name": f"func_{cwe_id}_{index}",
                    "cve_id": f"CVE-2026-{1000 + index}",
                    "commit_message": f"fix {cwe_id} sample {index}",
                }
            )

    rows.append(
        {
            "id": "119-0",
            "commit_id": "commit-119-0",
            "cwe_id": "CWE-119",
            "func_before": "int duplicate_short() {\n  return 0;\n}\n",
            "func_after": "int duplicate_short_fixed() {\n  return 1;\n}\n",
            "project": "duplicate-project",
        }
    )
    rows.append(
        {
            "id": "invalid-missing-after",
            "commit_id": "invalid-missing-after",
            "cwe_id": "CWE-119",
            "func_before": "int bad() { return 0; }\n",
            "func_after": "",
        }
    )
    rows.append(
        {
            "id": "invalid-unsupported-cwe",
            "commit_id": "invalid-unsupported-cwe",
            "cwe_id": "CWE-787",
            "func_before": "int bad2() { return 0; }\n",
            "func_after": "int bad2_fixed() { return 1; }\n",
        }
    )
    return rows


def write_config(path: Path, dataset_path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "model_name: gpt2",
                "device: cpu",
                "seed: 11",
                f"dataset_path: {dataset_path}",
                "target_cwes:",
                "  - CWE-119",
                "  - CWE-20",
                "pairs_per_cwe: 100",
                "split_mode: by_cwe",
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


def test_bigvul_preprocessing_filters_normalizes_and_deduplicates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "bigvul" / "raw"
    raw_file = dataset_path / "export.jsonl"
    write_bigvul_jsonl(raw_file, build_raw_rows(per_cwe=105))

    config = make_config(dataset_path, tmp_path / "outputs")
    manifest = preprocess_bigvul_dataset(config)

    normalized_rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "bigvul" / "processed" / "normalized.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["counts_by_cwe"] == {"CWE-119": 105, "CWE-20": 105}
    assert all(row["cwe_id"] in {"CWE-119", "CWE-20"} for row in normalized_rows)
    duplicate_row = next(row for row in normalized_rows if row["sample_id"] == "119-0")
    assert "duplicate_short" not in duplicate_row["vulnerable_code"]
    assert len([row for row in normalized_rows if row["sample_id"] == "119-0"]) == 1


def test_bigvul_sampling_is_reproducible_and_exact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "bigvul" / "raw"
    raw_file = dataset_path / "export.jsonl"
    write_bigvul_jsonl(raw_file, build_raw_rows(per_cwe=105))

    config = make_config(dataset_path, tmp_path / "outputs", pairs_per_cwe=100, seed=9)
    preprocess_bigvul_dataset(config)
    first = build_bigvul_dataset("cwe_119", config)
    second = build_bigvul_dataset("cwe_119", config)
    other = build_bigvul_dataset("cwe_20", config)

    assert len(first) == 100
    assert len(other) == 100
    assert first == second


def test_bigvul_preprocessing_fails_when_a_cwe_is_underfilled(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "bigvul" / "raw"
    raw_file = dataset_path / "export.jsonl"
    write_bigvul_jsonl(raw_file, build_raw_rows(per_cwe=2))

    config = make_config(dataset_path, tmp_path / "outputs", pairs_per_cwe=3)
    with pytest.raises(ValueError, match="Not enough cleaned Big-Vul pairs"):
        preprocess_bigvul_dataset(config)


def test_bigvul_task_registration_and_dataset_loading(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "bigvul" / "raw"
    raw_file = dataset_path / "export.jsonl"
    write_bigvul_jsonl(raw_file, build_raw_rows(per_cwe=105))

    config = make_config(dataset_path, tmp_path / "outputs")
    task = get_task("bigvul")

    assert isinstance(task, BigVulTask)
    assert task.split_names(config) == ["cwe_119", "cwe_20"]

    dataset = task.build_dataset("cwe_119", config)
    assert len(dataset) == 100
    example = dataset[0]
    assert example.cwe_id == "CWE-119"
    assert example.vulnerable_code
    assert example.patched_code
    assert example.split == "cwe_119"


def test_bigvul_behavior_and_experiment_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "bigvul" / "raw"
    raw_file = dataset_path / "export.jsonl"
    write_bigvul_jsonl(raw_file, build_raw_rows(per_cwe=105))

    config_path = tmp_path / "bigvul.yaml"
    output_dir = tmp_path / "outputs"
    write_config(config_path, dataset_path, output_dir)

    monkeypatch.setattr("mechinterp.core.model._import_transformer_bridge", lambda: FakeTransformerBridge)
    monkeypatch.setattr("mechinterp.core.model.ModelWrapper.forward_logits", fake_forward_logits)

    behavior_payload = run_behavior("bigvul", str(config_path))
    assert behavior_payload["task"] == "bigvul"
    assert set(behavior_payload["splits"]) == {"cwe_119", "cwe_20"}
    assert len(behavior_payload["results"]) == 200
    assert len(behavior_payload["all_results"]) == 400
    assert {row["pair_role"] for row in behavior_payload["all_results"]} == {"vulnerable", "patched"}
    assert {row["prediction"].strip() for row in behavior_payload["all_results"]} == {"yes", "no"}
    assert (output_dir / "bigvul" / "bigvul" / "behavior" / "results.json").exists()

    saved_payload = read_json(output_dir / "bigvul" / "bigvul" / "behavior" / "results.json")
    assert saved_payload["classification_summary"]["accuracy"] == 1.0

    cache_payload = run_cache("bigvul", str(config_path))
    assert cache_payload["task"] == "bigvul"
    assert cache_payload["selected_examples"]

    analysis_payload = run_analyze("bigvul", str(config_path))
    assert analysis_payload["task"] == "bigvul"
    assert "matched_pairs" in analysis_payload

    probe_payload = run_probe("bigvul", str(config_path))
    assert probe_payload["task"] == "bigvul"
    assert probe_payload["num_examples"] > 0

    ablation_payload = run_ablation("bigvul", str(config_path))
    assert ablation_payload["task"] == "bigvul"
    assert "results" in ablation_payload

    patch_payload = run_patch("bigvul", str(config_path))
    assert patch_payload["task"] == "bigvul"
    assert "aggregate" in patch_payload
