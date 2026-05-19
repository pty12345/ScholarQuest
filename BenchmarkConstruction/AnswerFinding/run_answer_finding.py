from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

try:
    from .answer_finding_pipeline import AnswerFindingPipeline
except ImportError:
    from answer_finding_pipeline import AnswerFindingPipeline


ANSWER_FINDING_ROOT = Path(__file__).resolve().parent
load_dotenv(ANSWER_FINDING_ROOT / ".env")


def _next_log_path(output_dir: Path) -> Path:
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for path in log_dir.glob("answer_*.json"):
        suffix = path.stem.removeprefix("answer_")
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return log_dir / f"answer_{max_idx + 1}.json"


async def _run(query: str, output_path: Path) -> None:
    pipeline = AnswerFindingPipeline()
    try:
        result = await pipeline.run(query)
    finally:
        await pipeline.close()

    payload = {"papers": asdict(result)["papers"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    log_path = _next_log_path(output_path.parent)
    log_payload = {
        "query": result.query,
        "logs": [asdict(item) for item in result.logs],
    }
    log_path.write_text(
        json.dumps(log_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the answer finding pipeline.")
    parser.add_argument(
        "query",
        nargs="?",
        default="RAG在医疗时序中的作用",
        type=str,
        help="The user query to expand and search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "answer_finding_results.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.query, args.output.resolve()))


if __name__ == "__main__":
    main()
