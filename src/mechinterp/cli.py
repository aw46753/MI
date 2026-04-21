"""Small CLI for mechinterp experiments."""

from __future__ import annotations

import argparse

from mechinterp.experiments.run_behavior import run as run_behavior
from mechinterp.experiments.run_cache import run as run_cache
from mechinterp.experiments.run_patching import run as run_patch
from mechinterp.experiments.summarize_ioi import run as run_summarize


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="mechinterp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("behavior", "cache", "patch", "summarize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--task", required=True)
        subparser.add_argument("--config", required=True)

    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "behavior":
        run_behavior(args.task, args.config)
    elif args.command == "cache":
        run_cache(args.task, args.config)
    elif args.command == "patch":
        run_patch(args.task, args.config)
    elif args.command == "summarize":
        run_summarize(args.task, args.config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
