from __future__ import annotations

import argparse
from pathlib import Path

import _add_src_to_path  # noqa: F401

from paperbench.io_utils import read_model_list
from paperbench.pipeline.runner import QueryPipelineRunner
from paperbench.types import TopicSeed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the query pool construction pipeline.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Optional path to a seed file in JSON or JSONL format.",
    )
    args = parser.parse_args()

    runner = QueryPipelineRunner.from_root(args.root.resolve())
    seeds = None
    if args.seed_file is not None:
        seed_path = args.seed_file.resolve()
        seeds = read_model_list(seed_path, TopicSeed)
    runner.run(seeds=seeds)


if __name__ == "__main__":
    main()
