from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .answer_finding_pipeline import AnswerFindingPipeline
except ImportError:
    from answer_finding_pipeline import AnswerFindingPipeline


ROOT = Path(__file__).resolve().parents[1]
ANSWER_FINDING_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "QueryGeneration" / "output" / "pasa_batch_results_flat.jsonl"
DEFAULT_RESULT_DIR = ROOT / "QueryGeneration" / "output" / "pasa_batch_results"
DEFAULT_OUTPUT_DIR = ANSWER_FINDING_ROOT / "output" / "pasa_batch_answer_finding"
load_dotenv(ANSWER_FINDING_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer finding for generated PASA batch queries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Flattened PASA query JSONL path.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Optional raw DashScope batch result directory. If set, records are flattened in memory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for answer outputs.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based record offset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process.")
    parser.add_argument("--resume", action="store_true", help="Skip query IDs already present in counts or errors.")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="Minimum paper score counted as an answer.")
    parser.add_argument(
        "--selector-threshold",
        type=float,
        default=0.05,
        help="Minimum selector score for keeping papers inside the answer finding pipeline.",
    )
    parser.add_argument(
        "--expand-threshold",
        type=float,
        default=0.1,
        help="Minimum score for iterative expansion after the initial search-paper expansion.",
    )
    parser.add_argument("--queries-per-round", type=int, default=10, help="Number of rewritten search anchors.")
    parser.add_argument("--search-top-k", type=int, default=10, help="Search results per rewritten query.")
    parser.add_argument("--citations-limit", type=int, default=30, help="Citation expansion limit per seed paper.")
    parser.add_argument("--references-limit", type=int, default=-1, help="Reference expansion limit per seed paper.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_flat_records(input_path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(input_path)
    for index, record in enumerate(records, start=1):
        query = record.get("final_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing final_query in {input_path}:{index}")
        if not isinstance(record.get("query_id"), str) or not str(record.get("query_id")).strip():
            record["query_id"] = f"BQ_{index:06d}"
    return records


def _load_result_dir_records(result_dir: Path) -> list[dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from QueryGeneration.parse_pasa_batch_results import _flatten_success_row, _read_jsonl as read_batch_jsonl

    records: list[dict[str, Any]] = []
    next_query_index = 1
    for output_file in sorted(result_dir.glob("*_output.jsonl")):
        for row in read_batch_jsonl(output_file):
            row_records, next_query_index = _flatten_success_row(
                row,
                source_file=output_file,
                next_query_index=next_query_index,
            )
            records.extend(row_records)
    if not records:
        raise FileNotFoundError(f"No query records found in {result_dir}")
    return records


def _slice_records(records: list[dict[str, Any]], *, start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    selected = records[start:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        selected = selected[:limit]
    return selected


def _processed_query_ids(counts_path: Path, errors_path: Path) -> set[str]:
    processed: set[str] = set()
    for row in _read_jsonl(counts_path) + _read_jsonl(errors_path):
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id:
            processed.add(query_id)
    return processed


def _filter_papers(papers: list[dict[str, Any]], score_threshold: float) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for paper in papers:
        score = paper.get("score")
        if isinstance(score, (int, float)) and float(score) > score_threshold:
            filtered.append(paper)
    return filtered


async def _run_one(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    query = str(record["final_query"]).strip()
    start_time = time.monotonic()
    pipeline = AnswerFindingPipeline(
        queries_per_round=args.queries_per_round,
        search_top_k=args.search_top_k,
        selector_threshold=args.selector_threshold,
        expand_threshold=args.expand_threshold,
        citations_limit=args.citations_limit,
        references_limit=args.references_limit,
    )
    try:
        result = await pipeline.run(query)
    finally:
        await pipeline.close()

    elapsed_seconds = round(time.monotonic() - start_time, 3)
    result_payload = asdict(result)
    papers = _filter_papers(result_payload["papers"], args.score_threshold)
    base_row = {
        "query_id": record["query_id"],
        "final_query": query,
        "answer_count": len(papers),
        "scored_paper_count": result_payload["scored_paper_count"],
        "score_filter": f"> {args.score_threshold}",
        "status": "ok",
        "elapsed_seconds": elapsed_seconds,
    }
    detail_row = {
        **base_row,
        "input_record": record,
        "first_round_queries": result_payload["first_round_queries"],
        "second_round_queries": result_payload["second_round_queries"],
        "logs": result_payload["logs"],
        "papers": papers,
    }
    return base_row, detail_row


def _write_summary(
    *,
    summary_path: Path,
    input_path: Path | None,
    result_dir: Path | None,
    query_count: int,
    records: list[dict[str, Any]],
    counts_path: Path,
    details_path: Path,
    errors_path: Path,
    score_threshold: float,
) -> None:
    completed = [row for row in records if row.get("status") == "ok"]
    errors = [row for row in records if row.get("status") == "error"]
    summary = {
        "input_path": str(input_path.resolve()) if input_path is not None else None,
        "result_dir": str(result_dir.resolve()) if result_dir is not None else None,
        "query_count": query_count,
        "processed_count": len(records),
        "completed_count": len(completed),
        "error_count": len(errors),
        "score_filter": f"> {score_threshold}",
        "total_answer_count": sum(int(row.get("answer_count") or 0) for row in completed),
        "total_scored_paper_count": sum(int(row.get("scored_paper_count") or 0) for row in completed),
        "counts_path": str(counts_path.resolve()),
        "details_path": str(details_path.resolve()),
        "errors_path": str(errors_path.resolve()),
        "records": records,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


async def main_async() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    counts_path = output_dir / "answer_counts.jsonl"
    details_path = output_dir / "answer_details.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "answer_counts_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.result_dir is not None:
        all_records = _load_result_dir_records(args.result_dir.resolve())
        input_path = None
        result_dir = args.result_dir.resolve()
    else:
        input_path = args.input.resolve()
        all_records = _load_flat_records(input_path)
        result_dir = None

    selected_records = _slice_records(all_records, start=args.start, limit=args.limit)
    skipped_ids = _processed_query_ids(counts_path, errors_path) if args.resume else set()
    existing_rows = _read_jsonl(counts_path)
    run_rows: list[dict[str, Any]] = []

    for index, record in enumerate(selected_records, start=1):
        query_id = str(record["query_id"])
        if query_id in skipped_ids:
            print(f"Skipping processed query {query_id}")
            continue

        print(f"Processing {index}/{len(selected_records)}: {query_id}")
        start_time = time.monotonic()
        try:
            count_row, detail_row = await _run_one(record, args)
            _append_jsonl(counts_path, count_row)
            if detail_row is not None:
                _append_jsonl(details_path, detail_row)
            run_rows.append(count_row)
            print(f"Completed {query_id}: {count_row['answer_count']} answers")
        except Exception as exc:
            elapsed_seconds = round(time.monotonic() - start_time, 3)
            error_row = {
                "query_id": query_id,
                "final_query": str(record.get("final_query", "")),
                "answer_count": None,
                "score_filter": f"> {args.score_threshold}",
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": elapsed_seconds,
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(counts_path, {key: value for key, value in error_row.items() if key != "traceback"})
            _append_jsonl(errors_path, error_row)
            run_rows.append({key: value for key, value in error_row.items() if key != "traceback"})
            print(f"Failed {query_id}: {exc}")

        _write_summary(
            summary_path=summary_path,
            input_path=input_path,
            result_dir=result_dir,
            query_count=len(selected_records),
            records=existing_rows + run_rows,
            counts_path=counts_path,
            details_path=details_path,
            errors_path=errors_path,
            score_threshold=args.score_threshold,
        )

    _write_summary(
        summary_path=summary_path,
        input_path=input_path,
        result_dir=result_dir,
        query_count=len(selected_records),
        records=existing_rows + run_rows,
        counts_path=counts_path,
        details_path=details_path,
        errors_path=errors_path,
        score_threshold=args.score_threshold,
    )
    print(f"Summary: {summary_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
