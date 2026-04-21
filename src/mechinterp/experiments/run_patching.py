"""Activation patching experiment entrypoint."""

from __future__ import annotations

from mechinterp.core.cache import build_names_filter
from mechinterp.core.hooks import make_residual_patch_hook
from mechinterp.core.metrics import final_token_logit_diff, normalized_patch_effect
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, run_dir, write_csv, write_json
from mechinterp.tasks.ioi.score import validate_single_token_candidate


def _layer_from_hook_name(hook_name: str) -> int:
    try:
        return int(hook_name.split(".")[1])
    except (IndexError, ValueError):
        return -1


def run(task_name: str, config_path: str) -> dict:
    """Run a residual-stream patch sweep over matched IOI pairs."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    model = ModelWrapper(config)
    hook_names = task.default_hook_names(config)
    names_filter = build_names_filter(hook_names)

    rows: list[dict] = []
    pair_count = 0

    for split in ("standard", "shifted"):
        dataset = task.build_dataset(split, config)
        scored_examples = [task.score_example(model, example) for example in dataset]
        pairs = task.make_pairs(dataset, scored_examples, model)

        for pair in pairs:
            pair_count += 1
            clean_tokens = model.to_tokens(pair.clean_prompt, prepend_bos=True)
            final_position = int(clean_tokens.shape[-1]) - 1
            correct_token_id = validate_single_token_candidate(model, pair.correct_token)
            wrong_token_id = validate_single_token_candidate(model, pair.wrong_token)

            _, clean_cache = model.run_with_cache(
                pair.clean_prompt,
                names_filter=names_filter,
                stop_at_layer=config.patch.max_layer + 1,
                return_type="logits",
                prepend_bos=True,
            )

            for hook_name in hook_names:
                patch_hook = make_residual_patch_hook(clean_cache, final_position)
                patched_logits = model.run_with_hooks(
                    pair.corrupted_prompt,
                    fwd_hooks=[(hook_name, patch_hook)],
                    return_type="logits",
                    prepend_bos=True,
                )
                patched_logit_diff = final_token_logit_diff(
                    patched_logits,
                    correct_token_id,
                    wrong_token_id,
                )
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "split": pair.split,
                        "template_id": pair.template_id,
                        "hook_name": hook_name,
                        "layer": _layer_from_hook_name(hook_name),
                        "clean_logit_diff": pair.clean_logit_diff,
                        "corrupted_logit_diff": pair.corrupted_logit_diff,
                        "patched_logit_diff": patched_logit_diff,
                        "normalized_effect": normalized_patch_effect(
                            pair.clean_logit_diff,
                            pair.corrupted_logit_diff,
                            patched_logit_diff,
                        ),
                    }
                )

    patch_dir = ensure_dir(run_dir(config, config_path) / "patch")
    write_csv(patch_dir / "results.csv", rows)
    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "pair_count": pair_count,
        "hook_names": hook_names,
        "results": rows,
    }
    write_json(patch_dir / "results.json", payload)
    return payload
