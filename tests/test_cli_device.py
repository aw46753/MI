import pytest

from mechinterp.cli import build_parser


@pytest.mark.parametrize(
    ("command", "device"),
    [
        ("behavior", "cuda"),
        ("cache", "cuda:0"),
        ("analyze", "cpu"),
        ("probe", "cuda"),
        ("ablate", "cuda:0"),
        ("patch", "cpu"),
        ("plot", "cuda"),
        ("summarize", "cuda:0"),
    ],
)
def test_cli_accepts_device_override_for_all_subcommands(command: str, device: str) -> None:
    parser = build_parser()

    args = parser.parse_args([command, "ioi", "--device", device])

    assert args.command == command
    assert args.task == "ioi"
    assert args.device == device
