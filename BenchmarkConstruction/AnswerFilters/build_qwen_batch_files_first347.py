from __future__ import annotations

import sys
from pathlib import Path

try:
    from .build_qwen_batch_files import main as build_qwen_batch_files_main
except ImportError:
    from build_qwen_batch_files import main as build_qwen_batch_files_main  # type: ignore[no-redef]


ANSWER_FILTER_ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = ANSWER_FILTER_ROOT.parent
DEFAULT_INPUT = (
    UPLOAD_ROOT
    / "AnswerFinding_v2"
    / "output"
    / "pasa_batch_answer_finding_qwen30b_test8"
    / "answer_details.jsonl"
)
DEFAULT_OUTPUT_DIR = ANSWER_FILTER_ROOT / "output" / "qwen_batch_inputs_first347"
DEFAULT_START = 0
DEFAULT_LIMIT = 347


def _has_option(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def _with_first347_defaults(argv: list[str]) -> list[str]:
    defaults: list[str] = []
    if not _has_option(argv, "--input"):
        defaults.extend(["--input", str(DEFAULT_INPUT)])
    if not _has_option(argv, "--output-dir"):
        defaults.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    if not _has_option(argv, "--start"):
        defaults.extend(["--start", str(DEFAULT_START)])
    if not _has_option(argv, "--limit"):
        defaults.extend(["--limit", str(DEFAULT_LIMIT)])
    return defaults + argv


def main() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *_with_first347_defaults(original_argv[1:])]
        build_qwen_batch_files_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
