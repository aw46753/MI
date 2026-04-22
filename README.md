# GPT-2 Error Analysis MechInterp Repo

This repository is a small, reusable starting point for mechanistic interpretability experiments on GPT-2 with TransformerLens. The current focus is not just whether GPT-2 gets a task right, but why it produces false positives and false negatives on structured tasks.

It currently includes:

- `ioi`: synthetic Indirect Object Identification with activation patching
- `addition`: synthetic two-digit addition with no-carry vs carry splits

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

Additional packages introduced for error-focused work:

- `evaluation/`: per-example prediction annotation and confusion-matrix metrics
- `analysis/`: error buckets, activation extraction, probes, matched pairs, ablations
- `plots/`: saved figures, including notebook-style Plotly heatmaps for intervention results

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
python -m mechinterp.cli behavior ioi
python -m mechinterp.cli analyze ioi
python -m mechinterp.cli cache ioi
python -m mechinterp.cli probe ioi
python -m mechinterp.cli ablate ioi
python -m mechinterp.cli patch ioi
python -m mechinterp.cli plot ioi
python -m mechinterp.cli summarize ioi

python -m mechinterp.cli behavior addition
python -m mechinterp.cli analyze addition
python -m mechinterp.cli cache addition
python -m mechinterp.cli probe addition
python -m mechinterp.cli ablate addition
python -m mechinterp.cli patch addition
python -m mechinterp.cli plot addition
python -m mechinterp.cli summarize addition
```

These default to `configs/<task>_small.yaml`. You can still override the config explicitly:

```bash
python -m mechinterp.cli behavior ioi configs/ioi_small.yaml
python -m mechinterp.cli behavior --task ioi --config configs/ioi_small.yaml
```

You can also use the console script after installation:

```bash
mechinterp behavior ioi
```

## What Each Command Does

- `behavior`: builds the requested task dataset, scores examples with final-token logit difference, annotates each example with `gold_label`, `predicted_label`, `confidence`, `margin`, and `error_type`, and writes CSV/JSON results.
- `analyze`: computes bucket summaries for `TP`, `TN`, `FP`, and `FN`, compares `FP vs TN` and `FN vs TP`, and builds matched error pairs for further interpretability work.
- `cache`: caches selected activations for scored examples using `names_filter` and `stop_at_layer`, then saves compact tensor artifacts and metadata.
- `probe`: extracts layerwise hidden states from `run_with_cache(...)` and fits simple linear probes per layer for:
  - all examples (`gold_label`)
  - `FN vs TP`
  - `FP vs TN`
- `ablate`: runs attention-head and MLP ablations on sampled examples from each error bucket and records the change in logit margin.
- `patch`: builds matched `FN -> TP` and `FP -> TN` pairs, patches `resid_pre` activations with `run_with_hooks(...)`, and writes layer-position patch effect grids.
- `plot`: generates confusion matrices, FPR/FNR subgroup bars, margin histograms, probe accuracy lines, patching heatmaps, and notebook-style ablation heatmaps when the corresponding inputs exist.
- `summarize`: reads saved outputs and prints a short text summary of behavior and patching results.

Outputs are written under `outputs/{task}/{run_name}/...`, for example:

```text
outputs/ioi/ioi_small/
outputs/addition/addition_small/
```

## FPR / FNR Analysis

`behavior` is the base experiment for false-positive / false-negative analysis. Each saved example includes:

- `gold_label`
- `predicted_label`
- `confidence`
- `margin`
- `error_type` (`TP`, `TN`, `FP`, `FN`)
- task metadata such as split, template, names, or arithmetic operands

The repository computes:

- accuracy
- precision
- recall
- FPR
- FNR

for the full task and separately by subgroup:

- IOI: `standard` vs `shifted`
- Addition: `standard` (no carry) vs `shifted` (carry)

## Interpretability Workflows

Typical workflows:

```bash
python -m mechinterp.cli behavior ioi
python -m mechinterp.cli analyze ioi
python -m mechinterp.cli probe ioi
python -m mechinterp.cli ablate ioi
python -m mechinterp.cli patch ioi
python -m mechinterp.cli plot ioi
```

and

```bash
python -m mechinterp.cli behavior addition
python -m mechinterp.cli analyze addition
python -m mechinterp.cli probe addition
python -m mechinterp.cli ablate addition
python -m mechinterp.cli patch addition
python -m mechinterp.cli plot addition
```

The intended interpretation is:

- `analyze`: identify where the model is overpredicting positives (`FP`) or missing positives (`FN`)
- `probe`: test whether class/error information is linearly decodable at each layer
- `ablate`: test whether specific heads or MLP blocks drive the problematic margins
- `patch`: test whether donor activations from matched correct examples can repair erroneous predictions

The ablation plots follow the TransformerLens demo notebook pattern for intervention results:

- interactive Plotly heatmaps
- zero-centered diverging color scale
- layer on the y-axis and head / bucket on the x-axis

## Adding a New Task Later

To add a new task, create a new folder under `src/mechinterp/tasks/` and implement the task contract from `src/mechinterp/tasks/base.py`:

- `split_names(config)`
- `build_behavior_split(model, split, config)`
- `build_dataset(split, config)`
- `score_example(model, example)`
- `make_pairs(dataset, scored_examples, model)`
- `build_error_pairs(rows, source_error_type, target_error_type, model)`
- `default_hook_names()`

The goal is that new tasks plug into the existing CLI and experiment runners without needing changes in `core/`.
