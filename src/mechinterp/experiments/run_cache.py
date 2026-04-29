"""Activation cache experiment entrypoint."""

from __future__ import annotations

from mechinterp.core.cache import build_names_filter, compact_cache, save_activation_artifacts, select_records
from mechinterp.core.model import ModelWrapper
from mechinterp.core.runner import get_task, load_experiment_config, log_progress, read_json, run_dir
from mechinterp.experiments.run_behavior import run as run_behavior


def run(task_name: str, config_path: str, *, device: str | None = None) -> dict:
    """Cache selected activations for scored examples."""

    config = load_experiment_config(config_path, device=device)
    task = get_task(task_name)
    if not task.supports_cache():
        raise ValueError(f"Caching is not implemented for task '{task_name}' yet.")

    behavior_path = run_dir(config, config_path, task_name=task_name) / "behavior" / "results.json"
    if behavior_path.exists():
        behavior_payload = read_json(behavior_path)
    else:
        behavior_payload = run_behavior(task_name, config_path, device=device)

    records = select_records(list(behavior_payload["all_results"]), config.cache.cache_num_examples)
    names_filter = build_names_filter(config.cache.cache_hook_names)
    model = ModelWrapper(config)
    log_progress(f"[cache] task={task_name} selected_examples={len(records)}")

    activations: dict[str, dict] = {}
    selected_metadata: list[dict] = []

    for index, row in enumerate(records):
        log_progress(f"[cache] caching example {index + 1}/{len(records)} split={row['split']}")
        prompt = row["prompt"]
        _, cache = model.run_with_cache(
            prompt,
            names_filter=names_filter,
            stop_at_layer=config.cache.stop_at_layer,
            return_type="logits",
            prepend_bos=True,
        )
        example_id = f"example_{index}"
        activations[example_id] = compact_cache(cache)
        selected_metadata.append(
            {
                "example_id": example_id,
                "prompt": prompt,
                "split": row["split"],
                "template_id": row.get("template_id"),
                "error_type": row.get("error_type"),
            }
        )

    cache_dir = run_dir(config, config_path, task_name=task_name) / "cache"
    metadata = {
        "task": task_name,
        "model_name": config.model_name,
        "hook_names": config.cache.cache_hook_names,
        "stop_at_layer": config.cache.stop_at_layer,
        "selected_examples": selected_metadata,
    }
    save_activation_artifacts(
        cache_dir / "activations.pt",
        cache_dir / "metadata.json",
        activations=activations,
        metadata=metadata,
    )
    log_progress(f"[cache] wrote {cache_dir / 'metadata.json'}")
    return metadata
