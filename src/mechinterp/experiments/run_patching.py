"""Activation patching experiment entrypoint."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from mechinterp.core.cache import build_names_filter
from mechinterp.core.hooks import make_residual_patch_hook
from mechinterp.core.metrics import final_token_logit_diff, normalized_patch_effect
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import ensure_dir, get_task, load_experiment_config, log_progress, read_json, run_dir, write_csv, write_json
from mechinterp.evaluation.metrics import annotate_prediction_rows
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.tasks.ioi.score import validate_single_token_candidate


def _layer_from_hook_name(hook_name: str) -> int:
    try:
        return int(hook_name.split(".")[1])
    except (IndexError, ValueError):
        return -1


def _selected_positions(num_positions: int, position_mode: str) -> list[int]:
    if position_mode == "all":
        return list(range(num_positions))
    if position_mode == "final":
        return [num_positions - 1]
    raise ValueError(f"Unsupported patch position mode {position_mode!r}")


def _candidate_rows_for_pair_type(
    rows: list[dict],
    source_error_type: str,
    target_error_type: str,
    *,
    per_bucket_limit: int | None,
) -> list[dict]:
    if per_bucket_limit is None:
        return rows

    source_rows = [row for row in rows if row["error_type"] == source_error_type][:per_bucket_limit]
    target_rows = [row for row in rows if row["error_type"] == target_error_type][:per_bucket_limit]
    return source_rows + target_rows


def _aggregate_patch_rows(rows: list[dict]) -> dict:
    by_layer_position: dict[tuple[int, int], list[float]] = defaultdict(list)
    tokens_by_position: dict[int, list[str]] = defaultdict(list)

    max_layer = -1
    max_position = -1
    for row in rows:
        layer = int(row["layer"])
        position = int(row["position"])
        by_layer_position[(layer, position)].append(float(row["normalized_effect"]))
        tokens_by_position[position].append(str(row["token"]))
        max_layer = max(max_layer, layer)
        max_position = max(max_position, position)

    if max_layer < 0 or max_position < 0:
        return {
            "layer_labels": [],
            "position_labels": [],
            "mean_normalized_effect": [],
            "top_entries": [],
        }

    matrix: list[list[float | None]] = []
    for layer in range(max_layer + 1):
        row_values: list[float | None] = []
        for position in range(max_position + 1):
            values = by_layer_position.get((layer, position))
            row_values.append(mean(values) if values else None)
        matrix.append(row_values)

    position_labels = []
    for position in range(max_position + 1):
        token_counts: dict[str, int] = defaultdict(int)
        for token in tokens_by_position.get(position, []):
            token_counts[token] += 1
        if token_counts:
            label = max(token_counts.items(), key=lambda item: item[1])[0]
            position_labels.append(f"{position}:{label}")
        else:
            position_labels.append(str(position))

    top_entries = sorted(
        (
            {
                "layer": layer,
                "position": position,
                "mean_normalized_effect": mean(values),
            }
            for (layer, position), values in by_layer_position.items()
        ),
        key=lambda entry: entry["mean_normalized_effect"],
        reverse=True,
    )[:10]

    return {
        "layer_labels": list(range(max_layer + 1)),
        "position_labels": position_labels,
        "mean_normalized_effect": matrix,
        "top_entries": top_entries,
    }


def run(task_name: str, config_path: str) -> dict:
    """Run residual-stream patching over matched error pairs."""

    config = load_experiment_config(config_path)
    task = get_task(task_name)
    if not task.supports_patching():
        raise ValueError(f"Patching is not implemented for task '{task_name}' yet.")
    model = ModelWrapper(config)
    hook_names = task.default_hook_names(config)
    names_filter = build_names_filter(hook_names)

    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path)

    behavior_rows = annotate_prediction_rows(list(behavior_payload["all_results"]))
    rows: list[dict] = []
    all_pairs: list = []
    per_bucket_limit = None
    if config.patch.max_pairs is not None:
        per_bucket_limit = max(1, (config.patch.max_pairs + 1) // 2)
    log_progress(
        f"[patch] task={task_name} preparing pairs from {len(behavior_rows)} rows with per_bucket_limit={per_bucket_limit}"
    )

    for source_error_type, target_error_type in (("FN", "TP"), ("FP", "TN")):
        candidate_rows = _candidate_rows_for_pair_type(
            behavior_rows,
            source_error_type,
            target_error_type,
            per_bucket_limit=per_bucket_limit,
        )
        log_progress(
            f"[patch] matching {source_error_type}->{target_error_type} from {len(candidate_rows)} candidate rows"
        )
        pairs = task.build_error_pairs(candidate_rows, source_error_type, target_error_type, model)
        log_progress(f"[patch] built {len(pairs)} pairs for {source_error_type}->{target_error_type}")
        all_pairs.extend(pairs)

    total_pairs = len(all_pairs)
    if config.patch.max_pairs is not None and total_pairs > config.patch.max_pairs:
        log_progress(f"[patch] limiting pairs from {total_pairs} to {config.patch.max_pairs}")
        all_pairs = all_pairs[: config.patch.max_pairs]

    pair_count = len(all_pairs)
    log_progress(
        f"[patch] task={task_name} pairs={pair_count} layers={len(hook_names)} position_mode={config.patch.position_mode}"
    )

    for pair_index, pair in enumerate(all_pairs, start=1):
        if pair_index == 1 or pair_index == pair_count or pair_index % 5 == 0:
            log_progress(f"[patch] processing pair {pair_index}/{pair_count} split={pair.split}")
        source_tokens = model.to_tokens(pair.source_prompt, prepend_bos=True)
        source_str_tokens = model.to_str_tokens(pair.source_prompt, prepend_bos=True)
        patch_positions = _selected_positions(
            int(source_tokens.shape[-1]),
            config.patch.position_mode,
        )
        final_position = int(source_tokens.shape[-1]) - 1
        correct_token_id = validate_single_token_candidate(model, pair.correct_token)
        wrong_token_id = validate_single_token_candidate(model, pair.wrong_token)

        _, clean_cache = model.run_with_cache(
            pair.target_prompt,
            names_filter=names_filter,
            stop_at_layer=config.patch.max_layer + 1,
            return_type=None,
            prepend_bos=True,
        )
        target_logits = model.forward_logits(pair.target_prompt, prepend_bos=True)
        target_logit_diff = final_token_logit_diff(
            target_logits,
            correct_token_id,
            wrong_token_id,
        )

        source_prediction_positive = pair.source_logit_diff > 0.0

        for hook_name in hook_names:
            layer = _layer_from_hook_name(hook_name)
            for position in patch_positions:
                patch_hook = make_residual_patch_hook(clean_cache, position)
                patched_logits = model.run_with_hooks(
                    pair.source_prompt,
                    fwd_hooks=[(hook_name, patch_hook)],
                    return_type="logits",
                    prepend_bos=True,
                )
                patched_logit_diff = final_token_logit_diff(
                    patched_logits,
                    correct_token_id,
                    wrong_token_id,
                )
                patched_prediction_positive = patched_logit_diff > 0.0
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "split": pair.split,
                        "source_error_type": pair.source_error_type,
                        "target_error_type": pair.target_error_type,
                        "hook_name": hook_name,
                        "layer": layer,
                        "position": position,
                        "token": source_str_tokens[position],
                        "final_position": final_position,
                        "target_logit_diff": target_logit_diff,
                        "source_logit_diff": pair.source_logit_diff,
                        "patched_logit_diff": patched_logit_diff,
                        "normalized_effect": normalized_patch_effect(
                            target_logit_diff,
                            pair.source_logit_diff,
                            patched_logit_diff,
                        ),
                        "prediction_flipped": source_prediction_positive != patched_prediction_positive,
                        "became_correct": (
                            patched_prediction_positive
                            if pair.source_error_type == "FN"
                            else not patched_prediction_positive
                        ),
                        **pair.metadata,
                    }
                )

    patch_dir = ensure_dir(run_dir(config, config_path, task_name=task_name) / "patch")
    write_csv(patch_dir / "results.csv", rows)
    aggregate = _aggregate_patch_rows(rows)
    payload = {
        "task": task_name,
        "model_name": config.model_name,
        "pair_count": pair_count,
        "pair_limit": config.patch.max_pairs,
        "hook_names": hook_names,
        "position_mode": config.patch.position_mode,
        "aggregate": aggregate,
        "results": rows,
    }
    write_json(patch_dir / "results.json", payload)
    log_progress(f"[patch] wrote {patch_dir / 'results.json'}")
    return payload
