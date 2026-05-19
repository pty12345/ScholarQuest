from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from .prompts import STRICT_BATCH_SYSTEM_PROMPT, build_strict_batch_user_prompt
    from .run_strict_cross_filter import DEFAULT_INPUT, _extract_compact_papers
except ImportError:
    from prompts import STRICT_BATCH_SYSTEM_PROMPT, build_strict_batch_user_prompt  # type: ignore[no-redef]
    from run_strict_cross_filter import DEFAULT_INPUT, _extract_compact_papers  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnswerFinding_v2.paper_agent_v2.utils import PaperSearchV2Client  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "qwen_batch_inputs"
DEFAULT_MODEL = "qwen-max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen batch JSONL files for strict answer filtering.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Compact answer details JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for batch files.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based query offset.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of query rows to convert.")
    parser.add_argument("--file-count", type=int, default=10, help="Number of batch JSONL files to write.")
    parser.add_argument("--papers-per-request", type=int, default=10, help="Papers included in one Qwen request.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name written into each request body.")
    parser.add_argument("--metadata-concurrency", type=int, default=32, help="Concurrent paper metadata requests.")
    parser.add_argument("--max-abstract-chars", type=int, default=4000, help="Maximum abstract chars per paper.")
    return parser.parse_args()


def _read_query_rows(path: Path, *, start: int, limit: int) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    if limit < 1:
        raise ValueError("--limit must be at least 1.")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < start:
                continue
            if len(rows) >= limit:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_index + 1}")
            rows.append(payload)
    return rows


async def _fetch_metadata(
    *,
    arxiv_ids: list[str],
    concurrency: int,
    max_abstract_chars: int,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    paper_client = PaperSearchV2Client(timeout=30.0, max_detail_concurrency=concurrency)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0

    async def fetch_one(arxiv_id: str) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                paper = await paper_client.get_paper(arxiv_id)
            except Exception as exc:
                return arxiv_id, {
                    "found": False,
                    "title": "",
                    "abstract": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if paper is None or not paper.title or not paper.abstract:
                return arxiv_id, {"found": False, "title": "", "abstract": ""}
            abstract = paper.abstract
            if max_abstract_chars > 0 and len(abstract) > max_abstract_chars:
                abstract = abstract[:max_abstract_chars].rstrip()
            return arxiv_id, {"found": True, "title": paper.title, "abstract": abstract}

    try:
        tasks = [asyncio.create_task(fetch_one(arxiv_id)) for arxiv_id in arxiv_ids]
        for future in asyncio.as_completed(tasks):
            arxiv_id, metadata = await future
            cache[arxiv_id] = metadata
            completed += 1
            if completed % 500 == 0 or completed == len(arxiv_ids):
                print(f"Fetched metadata: {completed}/{len(arxiv_ids)}", flush=True)
    finally:
        await paper_client.close()

    return cache


def _candidate_for_prompt(candidate: dict[str, Any], metadata: dict[str, Any], paper_index: int) -> dict[str, Any]:
    return {
        "paper_index": paper_index,
        "arxiv_id": candidate["arxiv_id"],
        "title": str(metadata["title"]),
        "abstract": str(metadata["abstract"]),
    }


def _build_request(
    *,
    custom_id: str,
    model: str,
    query: str,
    prompt_papers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": STRICT_BATCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_strict_batch_user_prompt(query=query, papers=prompt_papers),
                },
            ],
            "temperature": 0.1,
        },
    }


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


async def main_async() -> None:
    args = parse_args()
    if args.file_count < 1:
        raise ValueError("--file-count must be at least 1.")
    if args.papers_per_request < 1:
        raise ValueError("--papers-per-request must be at least 1.")
    if args.metadata_concurrency < 1:
        raise ValueError("--metadata-concurrency must be at least 1.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_query_rows(args.input.resolve(), start=args.start, limit=args.limit)
    query_items: list[dict[str, Any]] = []
    arxiv_ids: list[str] = []
    seen_arxiv_ids: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id") or "").strip()
        query = str(row.get("final_query") or "").strip()
        if not query_id or not query:
            continue
        candidates = _extract_compact_papers(row)
        query_items.append({"query_id": query_id, "query": query, "candidates": candidates})
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id") or "").strip()
            if arxiv_id and arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                arxiv_ids.append(arxiv_id)

    metadata_cache = await _fetch_metadata(
        arxiv_ids=arxiv_ids,
        concurrency=args.metadata_concurrency,
        max_abstract_chars=args.max_abstract_chars,
    )

    requests: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    dropped_candidate_count = 0
    for query_item in query_items:
        query_id = query_item["query_id"]
        query = query_item["query"]
        candidates_with_metadata: list[dict[str, Any]] = []
        for candidate in query_item["candidates"]:
            arxiv_id = str(candidate.get("arxiv_id") or "")
            metadata = metadata_cache.get(arxiv_id)
            if metadata is None or not bool(metadata.get("found")):
                dropped_candidate_count += 1
                continue
            candidates_with_metadata.append(candidate)

        for batch_index, candidate_batch in enumerate(_chunks(candidates_with_metadata, args.papers_per_request), start=1):
            prompt_papers = [
                _candidate_for_prompt(
                    candidate,
                    metadata_cache[str(candidate["arxiv_id"])],
                    paper_index=batch_offset,
                )
                for batch_offset, candidate in enumerate(candidate_batch, start=1)
            ]
            custom_id = f"{query_id}__qwen_batch_{batch_index:04d}"
            requests.append(
                _build_request(
                    custom_id=custom_id,
                    model=args.model,
                    query=query,
                    prompt_papers=prompt_papers,
                )
            )
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "query_id": query_id,
                    "final_query": query,
                    "batch_index": batch_index,
                    "paper_count": len(candidate_batch),
                    "papers": [
                        {
                            "paper_index": index,
                            "arxiv_id": candidate["arxiv_id"],
                            "previous_stage_score": candidate.get("original_score"),
                            "previous_stage_confidence": candidate.get("original_confidence"),
                        }
                        for index, candidate in enumerate(candidate_batch, start=1)
                    ],
                }
            )

    requests_per_file = math.ceil(len(requests) / args.file_count) if requests else 0
    batch_files: list[dict[str, Any]] = []
    for file_index in range(args.file_count):
        start = file_index * requests_per_file if requests_per_file else 0
        end = start + requests_per_file if requests_per_file else 0
        file_requests = requests[start:end]
        file_path = output_dir / f"qwen_batch_requests_{file_index + 1:02d}.jsonl"
        _write_jsonl(file_path, file_requests)
        batch_files.append({"path": str(file_path), "request_count": len(file_requests)})

    manifest_path = output_dir / "qwen_batch_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    summary = {
        "input_path": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "model": args.model,
        "query_count_requested": args.limit,
        "query_count_available": len(rows),
        "query_count_written": len(query_items),
        "candidate_count": sum(len(item["candidates"]) for item in query_items),
        "unique_paper_count": len(arxiv_ids),
        "dropped_candidate_count": dropped_candidate_count,
        "papers_per_request": args.papers_per_request,
        "request_count": len(requests),
        "file_count": args.file_count,
        "batch_files": batch_files,
        "manifest_path": str(manifest_path),
    }
    summary_path = output_dir / "qwen_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
