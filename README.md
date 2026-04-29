# GPT-2 Error Analysis MechInterp Repo

This repository is a small, reusable starting point for mechanistic interpretability experiments on GPT-2 with TransformerLens. The current focus is not just whether GPT-2 gets a task right, but why it produces false positives and false negatives on structured tasks.

It currently includes:

- `ioi`: synthetic Indirect Object Identification with activation patching
- `addition`: synthetic two-digit addition with no-carry vs carry splits
- `greater_than`: synthetic binary numeric comparison with easy vs small-gap splits
- `sva`: synthetic subject-verb agreement with attractor interference splits
- `bigvul`: paired vulnerable/patched C/C++ functions from Big-Vul for CWE-119 and CWE-20

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

This environment currently has a CUDA build of PyTorch installed (`torch 2.11.0+cu126`). If your shell/runtime exposes an NVIDIA GPU, you can run the experiments on GPU by passing a device override at the CLI.

## CLI Usage

Run the CLI via the package module:

```bash
python -m mechinterp.cli behavior ioi
python -m mechinterp.cli behavior ioi --device cuda
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

python -m mechinterp.cli behavior greater_than
python -m mechinterp.cli analyze greater_than
python -m mechinterp.cli cache greater_than
python -m mechinterp.cli probe greater_than
python -m mechinterp.cli ablate greater_than
python -m mechinterp.cli patch greater_than
python -m mechinterp.cli plot greater_than
python -m mechinterp.cli summarize greater_than

python -m mechinterp.cli behavior sva
python -m mechinterp.cli analyze sva
python -m mechinterp.cli cache sva
python -m mechinterp.cli probe sva
python -m mechinterp.cli ablate sva
python -m mechinterp.cli patch sva
python -m mechinterp.cli plot sva
python -m mechinterp.cli summarize sva

python -m mechinterp.cli behavior bigvul
python -m mechinterp.cli patch bigvul --device cuda:0
python -m mechinterp.cli analyze bigvul
python -m mechinterp.cli cache bigvul
python -m mechinterp.cli probe bigvul
python -m mechinterp.cli ablate bigvul
python -m mechinterp.cli patch bigvul
python -m mechinterp.cli plot bigvul
python -m mechinterp.cli summarize bigvul
```

These default to `configs/<task>_small.yaml`. You can still override the config explicitly:

```bash
python -m mechinterp.cli behavior ioi configs/ioi_small.yaml
python -m mechinterp.cli behavior --task ioi --config configs/ioi_small.yaml
python -m mechinterp.cli behavior ioi configs/ioi_small.yaml --device cuda
```

You can also use the console script after installation:

```bash
mechinterp behavior ioi
mechinterp behavior ioi --device cuda
```

The YAML `device` field still works, but `--device` takes precedence for that run. Supported values are `cpu`, `cuda`, and explicit CUDA devices such as `cuda:0`.

If you request `--device cuda` and no GPU is visible, the CLI now fails with a clear error instead of silently running on CPU.

## Run Everything On GPU

If you are already inside the repo-local virtualenv, run the CLI from source with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m mechinterp.cli behavior ioi --device cuda
```

If you prefer to call the interpreter explicitly without activating the virtualenv first, use:

```bash
PYTHONPATH=src ./.venv/bin/python -m mechinterp.cli behavior ioi --device cuda
```

Before launching experiments, verify that this Python environment can see your GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

To run the full workflow for a single task on GPU:

```bash
PYTHONPATH=src python -m mechinterp.cli behavior bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli analyze bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli cache bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli probe bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli ablate bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli patch bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli plot bigvul --device cuda
PYTHONPATH=src python -m mechinterp.cli summarize bigvul --device cuda
```

Replace `ioi` with any supported task:

- `addition`
- `greater_than`
- `sva`
- `bigvul`

To run the full workflow for every task on GPU:

```bash
for task in ioi addition greater_than sva bigvul; do
  PYTHONPATH=src python -m mechinterp.cli behavior "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli analyze "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli cache "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli probe "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli ablate "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli patch "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli plot "$task" --device cuda
  PYTHONPATH=src python -m mechinterp.cli summarize "$task" --device cuda
done
```

If you want a specific GPU, use an explicit device such as `--device cuda:0`.

If you later install the package into the virtualenv with `python -m pip install -e . --no-build-isolation`, you can drop `PYTHONPATH=src` and use `python -m mechinterp.cli ...` directly.

## GPU Troubleshooting

If PyTorch has CUDA support installed but `--device cuda` still fails:

- check that the runtime actually exposes an NVIDIA GPU to this shell
- confirm the NVIDIA driver is installed and working
- verify `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` reports an available device
- if you are running in a container, VM, WSL, or managed environment, make sure GPU passthrough is enabled for that session

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
outputs/greater_than/greater_than_small/
outputs/sva/sva_small/
outputs/bigvul/bigvul_small/
```

## Big-Vul Data Setup

`bigvul` expects a local cleaned dataset under `data/bigvul/raw/` in `.jsonl`, `.json`, or `.csv` format. The current pipeline expects actual function pairs rather than commit ids alone.

Each usable record should contain:

- `sample_id`
- `commit_id`
- `cwe_id`
- `func_before`
- `func_after`

Optional metadata fields that are preserved when present:

- `project`
- `file_path`
- `function_name`
- `cve_id`
- `commit_message`

The preprocessing pipeline:

- filters to `CWE-119` and `CWE-20`
- drops rows missing either function body
- normalizes CWE labels such as `119` to `CWE-119`
- deduplicates by stable sample id
- samples a fixed number of pairs per CWE using the config seed
- writes processed artifacts to `data/bigvul/processed/`

The checked-in config at `configs/bigvul_small.yaml` expects `100` usable pairs for each target CWE. For smaller validation files, lower `pairs_per_cwe` in an override config.

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
- Greater-than: `standard` (large gap) vs `shifted` (small gap)
- SVA: `standard` (no attractor) vs `shifted` (opposite-number attractor)
- Big-Vul: `cwe_119` vs `cwe_20`

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

and

```bash
python -m mechinterp.cli behavior greater_than
python -m mechinterp.cli analyze greater_than
python -m mechinterp.cli probe greater_than
python -m mechinterp.cli ablate greater_than
python -m mechinterp.cli patch greater_than
python -m mechinterp.cli plot greater_than
```

and

```bash
python -m mechinterp.cli behavior sva
python -m mechinterp.cli analyze sva
python -m mechinterp.cli probe sva
python -m mechinterp.cli ablate sva
python -m mechinterp.cli patch sva
python -m mechinterp.cli plot sva
```

and

```bash
python -m mechinterp.cli behavior bigvul
python -m mechinterp.cli analyze bigvul
python -m mechinterp.cli cache bigvul
python -m mechinterp.cli probe bigvul
python -m mechinterp.cli ablate bigvul
python -m mechinterp.cli patch bigvul
python -m mechinterp.cli plot bigvul
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

For `bigvul`, `behavior` uses a simple GPT-2 prompt of the form:

- instruction: `You are a software security expert.`
- question: `Is the following C/C++ function vulnerable to <CWE>?`
- answer constraint: `Answer only Yes or No.`
- code block containing `func_before` or `func_after`
- final-token scoring between the candidates ` yes` and ` no`

This lets the same cache / probe / ablation / patching pipeline run over Big-Vul prompts, although pair counts for patching may be smaller when token lengths differ across matched examples.

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
