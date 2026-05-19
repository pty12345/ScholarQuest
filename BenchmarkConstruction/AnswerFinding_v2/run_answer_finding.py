from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .answer_finding_pipeline import AnswerFindingPipeline
except ImportError:
    from answer_finding_pipeline import AnswerFindingPipeline  # type: ignore[no-redef]


ANSWER_FINDING_ROOT = Path(__file__).resolve().parent
load_dotenv(ANSWER_FINDING_ROOT / ".env")


def _include_output_paper(paper: dict[str, Any]) -> bool:
    level = int(paper.get("relevance_level", paper.get("score", 0)))
    confidence = str(paper.get("confidence") or "").strip().lower()
    return level >= 2 or (level == 1 and confidence in {"low", "medium"})


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

    result_payload = asdict(result)
    payload = {
        "weakly_relevant_count": result_payload["weakly_relevant_count"],
        "weak_low_medium_count": result_payload["weak_low_medium_count"],
        "papers": [
            paper
            for paper in result_payload["papers"]
            if _include_output_paper(paper)
        ],
    }
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
