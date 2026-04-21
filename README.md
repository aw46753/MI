# GPT-2 IOI MechInterp Starter Repo

This repository is a small, reusable starting point for mechanistic interpretability experiments on GPT-2 with TransformerLens. The initial scope is intentionally narrow: only the Indirect Object Identification (IOI) task is implemented, but the code is structured so additional tasks can be added later without rewriting the core experiment plumbing.

The implementation style follows the key TransformerLens demo primitives rather than building a notebook-only project:

- model loading with `TransformerBridge.boot_transformers(...)`
- activation caching with `run_with_cache(...)`
- bounded caching with `names_filter` and `stop_at_layer`
- hook-based interventions with `run_with_hooks(...)`

TransformerLens 3.x deprecates `HookedTransformer` for new code. This repo therefore uses `TransformerBridge` in the shared model wrapper so the code path matches current guidance and is easier to extend forward.

References:

- TransformerLens contributing guide: <https://transformerlensorg.github.io/TransformerLens/content/contributing.html>
- TransformerLens exploratory IOI demo: <https://transformerlensorg.github.io/TransformerLens/generated/demos/Exploratory_Analysis_Demo.html>

## Repository Layout

```text
configs/
src/mechinterp/
  core/
  tasks/
  experiments/
outputs/
tests/
```

`core/` contains reusable model, cache, hook, metric, pairing, and config helpers. `tasks/` contains task-specific logic. `experiments/` contains thin entrypoints used by the CLI.

## Installation

Create a virtual environment, then install either the exact tested pins or the package directly.

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e '.[dev]'
```

On Linux, the default PyPI `torch` wheel may pull large CUDA dependencies. If you want a smaller CPU-only setup, install PyTorch from the official selector first, then install this repo with `--no-deps`.

## CLI Usage

Run the CLI via the package module:

```bash
python -m mechinterp.cli behavior --task ioi --config configs/ioi_small.yaml
python -m mechinterp.cli cache --task ioi --config configs/ioi_small.yaml
python -m mechinterp.cli patch --task ioi --config configs/ioi_small.yaml
python -m mechinterp.cli summarize --task ioi --config configs/ioi_small.yaml
```

You can also use the console script after installation:

```bash
mechinterp behavior --task ioi --config configs/ioi_small.yaml
```

## What Each Command Does

- `behavior`: builds the IOI dataset, scores examples with final-token logit difference, and writes CSV and JSON results.
- `cache`: caches selected activations for scored examples using `names_filter` and `stop_at_layer`, then saves compact tensor artifacts and metadata.
- `patch`: builds matched clean/corrupted IOI pairs, patches residual stream activations with `run_with_hooks(...)`, and saves patch effect summaries.
- `summarize`: reads saved outputs and prints a short text summary of behavior and patching results.

Outputs are written under `outputs/ioi_small/` by default.

## Adding a New Task Later

To add a new task, create a new folder under `src/mechinterp/tasks/`, for example `tasks/addition/`, and implement the task contract from `src/mechinterp/tasks/base.py`:

- `build_dataset(split, config)`
- `score_example(model, example)`
- `make_pairs(dataset, scored_examples, model)`
- `default_hook_names()`

The goal is that new tasks plug into the existing CLI and experiment runners without needing changes in `core/`.
