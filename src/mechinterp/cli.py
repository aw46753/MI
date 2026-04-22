"""Small CLI for mechinterp experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from mechinterp.experiments.run_ablation import run as run_ablation
from mechinterp.experiments.run_error_analysis import run as run_analyze
from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.experiments.run_cache import run as run_cache
from mechinterp.experiments.run_patching import run as run_patch
from mechinterp.experiments.run_plots import run as run_plot
from mechinterp.experiments.run_probes import run as run_probe
from mechinterp.experiments.summarize_ioi import run as run_summarize


def _default_config_path(task: str) -> str:
    """Return the default config path for a task."""

    return str(Path("configs") / f"{task}_small.yaml")


def _resolve_task_and_config(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve task/config from positional args or legacy flags."""

    task = args.task or args.task_flag
    if task is None:
        raise ValueError("A task is required. Use e.g. `mechinterp behavior ioi`.")

    config = args.config or args.config_flag or _default_config_path(task)
    return task, config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="mechinterp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("behavior", "cache", "analyze", "probe", "ablate", "patch", "plot", "summarize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("task", nargs="?")
        subparser.add_argument("config", nargs="?")
        subparser.add_argument("--task", dest="task_flag")
        subparser.add_argument("--config", dest="config_flag")

    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        task, config = _resolve_task_and_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "behavior":
        run_behavior(task, config)
    elif args.command == "cache":
        run_cache(task, config)
    elif args.command == "analyze":
        run_analyze(task, config)
    elif args.command == "probe":
        run_probe(task, config)
    elif args.command == "ablate":
        run_ablation(task, config)
    elif args.command == "patch":
        run_patch(task, config)
    elif args.command == "plot":
        run_plot(task, config)
    elif args.command == "summarize":
        run_summarize(task, config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
