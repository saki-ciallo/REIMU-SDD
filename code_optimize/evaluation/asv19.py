from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Allow direct execution as ``python code_optimize/evaluation/asv19.py``.
CODE_OPTIMIZE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_OPTIMIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_OPTIMIZE_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASVspoof2019 test or scoring workflow.")
    parser.add_argument(
        "stage",
        choices=("test", "score"),
        help="Run inference or score existing ASV19 prediction files.",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected ASV19 command.",
    )
    arguments = parser.parse_args(argv)
    if arguments.arguments[:1] == ["--"]:
        arguments.arguments = arguments.arguments[1:]
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    from evaluation.asv19_score import main as score_main
    from evaluation.asv19_test import main as test_main

    arguments = parse_args(argv)
    if arguments.stage == "test":
        test_main(arguments.arguments)
    else:
        score_main(arguments.arguments)


if __name__ == "__main__":
    main()
