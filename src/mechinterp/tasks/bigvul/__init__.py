"""Big-Vul task package."""

from __future__ import annotations

from typing import Any

from mechinterp.analysis.matched_pairs import build_matched_pairs_from_groups
from mechinterp.core.hooks import resid_pre_hook_name
from mechinterp.tasks.base import Task
from mechinterp.tasks.bigvul.data import BigVulExample, build_bigvul_dataset, normalize_cwe_label
from mechinterp.tasks.ioi.score import IOIScoreResult, score_prompt_with_candidates


POSITIVE_TOKEN = " yes"
NEGATIVE_TOKEN = " no"


def render_bigvul_prompt(example: BigVulExample, code_text: str) -> str:
    """Render a simple GPT-2 classification prompt for one code snippet."""

    return (
        "You are a software security expert.\n\n"
        f"Is the following C/C++ function vulnerable to {example.cwe_id}?\n"
        "Answer only Yes or No.\n\n"
        "```c\n"
        f"{code_text}\n"
        "```\n\n"
        "Answer:"
    )


class BigVulTask(Task):
    """Big-Vul paired testcase task scored with GPT-2 prompts."""

    name = "bigvul"

    def split_names(self, config: Any) -> list[str]:
        splits = []
        for cwe_id in config.dataset.target_cwes:
            normalized = normalize_cwe_label(cwe_id)
            if normalized is None:
                raise ValueError(f"Unsupported Big-Vul CWE id: {cwe_id}")
            splits.append(normalized.lower().replace("-", "_"))
        return splits

    def build_behavior_split(self, model: Any, split: str, config: Any) -> dict[str, Any]:
        dataset = self.build_dataset(split, config)
        vulnerable_rows = []
        patched_rows = []
        all_rows = []

        for example in dataset:
            shared = {
                "sample_id": example.sample_id,
                "split": split,
                "cwe_id": example.cwe_id,
                "commit_id": example.commit_id,
                "project": example.project,
                "file_path": example.file_path,
                "function_name": example.function_name,
                "cve_id": example.cve_id,
                "commit_message": example.commit_message,
                "scoring_mode": "gpt2_candidate_logit_diff",
                "model_score_placeholder": False,
                "positive_token": POSITIVE_TOKEN,
                "negative_token": NEGATIVE_TOKEN,
            }

            vulnerable_prompt = render_bigvul_prompt(example, example.vulnerable_code)
            vulnerable_result = score_prompt_with_candidates(
                model,
                vulnerable_prompt,
                correct_token=POSITIVE_TOKEN,
                wrong_token=NEGATIVE_TOKEN,
                metadata={
                    **shared,
                    "prompt": vulnerable_prompt,
                    "code_text": example.vulnerable_code,
                    "code_line_count": example.vulnerable_line_count,
                    "pair_role": "vulnerable",
                    "expected_positive": True,
                    "correct_token": POSITIVE_TOKEN,
                    "wrong_token": NEGATIVE_TOKEN,
                },
            )
            vulnerable_row = {
                **vulnerable_result.to_flat_dict(),
                "code_text": example.vulnerable_code,
                "pair_role": "vulnerable",
                "expected_positive": True,
            }

            patched_prompt = render_bigvul_prompt(example, example.patched_code)
            patched_result = score_prompt_with_candidates(
                model,
                patched_prompt,
                correct_token=POSITIVE_TOKEN,
                wrong_token=NEGATIVE_TOKEN,
                metadata={
                    **shared,
                    "prompt": patched_prompt,
                    "code_text": example.patched_code,
                    "code_line_count": example.patched_line_count,
                    "pair_role": "patched",
                    "expected_positive": False,
                    "correct_token": POSITIVE_TOKEN,
                    "wrong_token": NEGATIVE_TOKEN,
                },
            )
            patched_row = {
                **patched_result.to_flat_dict(),
                "code_text": example.patched_code,
                "pair_role": "patched",
                "expected_positive": False,
            }
            vulnerable_rows.append(vulnerable_row)
            patched_rows.append(patched_row)
            all_rows.extend((vulnerable_row, patched_row))

        return {
            "primary_rows": vulnerable_rows,
            "all_rows": all_rows,
            "positive_rows": vulnerable_rows,
            "negative_rows": patched_rows,
            "positive_summary_name": "vulnerable",
            "negative_summary_name": "patched",
        }

    def build_dataset(self, split: str, config: Any) -> list[BigVulExample]:
        return build_bigvul_dataset(split, config)

    def score_example(self, model: Any, example: BigVulExample) -> IOIScoreResult:
        prompt = render_bigvul_prompt(example, example.vulnerable_code)
        return score_prompt_with_candidates(
            model,
            prompt,
            correct_token=POSITIVE_TOKEN,
            wrong_token=NEGATIVE_TOKEN,
            metadata={
                **example.to_dict(),
                "prompt": prompt,
                "code_text": example.vulnerable_code,
                "code_line_count": example.vulnerable_line_count,
                "pair_role": "vulnerable",
                "expected_positive": True,
                "positive_token": POSITIVE_TOKEN,
                "negative_token": NEGATIVE_TOKEN,
                "scoring_mode": "gpt2_candidate_logit_diff",
            },
        )

    def make_pairs(self, dataset: list[BigVulExample], scored_examples: list[Any], model: Any) -> list[Any]:
        return []

    def build_error_pairs(
        self,
        rows: list[dict[str, Any]],
        source_error_type: str,
        target_error_type: str,
        model: Any,
    ) -> list[Any]:
        source_rows = [row for row in rows if row["error_type"] == source_error_type]
        target_rows = [row for row in rows if row["error_type"] == target_error_type]
        return build_matched_pairs_from_groups(
            task_name=self.name,
            source_rows=source_rows,
            target_rows=target_rows,
            source_error_type=source_error_type,
            target_error_type=target_error_type,
            model=model,
            pair_score=lambda left, right: (
                0 if left.get("cwe_id") == right.get("cwe_id") else 10,
                0 if left.get("project") == right.get("project") else 1,
                abs(int(left.get("code_line_count", 0)) - int(right.get("code_line_count", 0))),
                abs(len(str(left.get("code_text", ""))) - len(str(right.get("code_text", "")))),
            ),
            metadata_keys=[
                "sample_id",
                "commit_id",
                "cwe_id",
                "project",
                "file_path",
                "function_name",
                "pair_role",
                "code_line_count",
            ],
        )

    def default_hook_names(self, config: Any) -> list[str]:
        return [resid_pre_hook_name(layer) for layer in range(config.patch.max_layer + 1)]
