from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from .prompts import STRICT_BATCH_SYSTEM_PROMPT, build_strict_batch_user_prompt
except ImportError:
    from prompts import STRICT_BATCH_SYSTEM_PROMPT, build_strict_batch_user_prompt  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AnswerFinding_v2.paper_agent_v2.utils import (  # noqa: E402
    Paper,
    PaperSearchV2Client,
    call_openai_chat,
    normalize_arxiv_id,
)


ANSWER_FILTER_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "AnswerFinding_v2"
    / "output"
    / "pasa_batch_answer_finding_qwen30b_test8"
    / "answer_details_compact.jsonl"
)
DEFAULT_OUTPUT_DIR = ANSWER_FILTER_ROOT / "output" / "strict_cross_filter"
load_dotenv(ANSWER_FILTER_ROOT / ".env")


@dataclass(slots=True)
class ModelConfig:
    label: str
    model: str
    api_base: str
    api_key: str


@dataclass(slots=True)
class PaperCandidate:
    paper_index: int
    arxiv_id: str
    title: str
    abstract: str
    original_score: Any = None
    original_confidence: str = ""


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    return int(raw_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict qwen/deepseek cross filtering.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Compact answer details JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based input row offset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of input rows.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all selected query IDs instead of skipping completed ones.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("ANSWER_FILTER_BATCH_SIZE", 10),
        help="Number of papers per judge request.",
    )
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=_env_int("ANSWER_FILTER_QUERY_CONCURRENCY", 1),
        help="Maximum number of queries processed concurrently.",
    )
    parser.add_argument(
        "--model-request-concurrency",
        type=int,
        default=_env_int("ANSWER_FILTER_MODEL_REQUEST_CONCURRENCY", 4),
        help="Maximum concurrent batch requests per model within each query.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=_env_int("ANSWER_FILTER_MAX_RETRIES", 2),
        help="Retries for each model batch before fallback.",
    )
    parser.add_argument(
        "--max-abstract-chars",
        type=int,
        default=_env_int("ANSWER_FILTER_MAX_ABSTRACT_CHARS", 4000),
        help="Maximum abstract characters sent to each judge.",
    )
    return parser.parse_args()


def _model_configs() -> tuple[ModelConfig, ModelConfig]:
    qwen = ModelConfig(
        label="qwen",
        model=os.getenv("ANSWER_FILTER_QWEN_MODEL", "qwen-max").strip(),
        api_base=os.getenv(
            "ANSWER_FILTER_QWEN_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip(),
        api_key=os.getenv("ANSWER_FILTER_QWEN_API_KEY", "").strip(),
    )
    deepseek = ModelConfig(
        label="deepseek",
        model=os.getenv("ANSWER_FILTER_DEEPSEEK_MODEL", "deepseek-chat").strip(),
        api_base=os.getenv("ANSWER_FILTER_DEEPSEEK_API_BASE", "https://api.deepseek.com").strip(),
        api_key=os.getenv("ANSWER_FILTER_DEEPSEEK_API_KEY", "").strip(),
    )
    return qwen, deepseek


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


def _slice_rows(rows: list[dict[str, Any]], *, start: int, limit: Optional[int]) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative.")
    selected = rows[start:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        selected = selected[:limit]
    return selected


def _completed_query_ids(counts_path: Path, details_path: Path) -> set[str]:
    completed: set[str] = set()
    for row in _read_jsonl(counts_path) + _read_jsonl(details_path):
        if row.get("status") != "ok":
            continue
        query_id = str(row.get("query_id") or "").strip()
        if query_id:
            completed.add(query_id)
    return completed


def _extract_json_object(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response does not contain a JSON object.")
    return cleaned[start : end + 1]


def _extract_compact_papers(row: dict[str, Any]) -> list[dict[str, Any]]:
    papers = row.get("papers")
    extracted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if isinstance(papers, list):
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            arxiv_id = normalize_arxiv_id(str(paper.get("arxiv_id") or ""))
            score = paper.get("score", paper.get("relevance_level"))
            if not arxiv_id or arxiv_id in seen_ids or not isinstance(score, (int, float)) or float(score) < 2:
                continue
            seen_ids.add(arxiv_id)
            extracted.append(
                {
                    "arxiv_id": arxiv_id,
                    "original_score": score,
                    "original_confidence": str(paper.get("confidence") or "").strip().lower(),
                }
            )
    if extracted:
        return extracted

    # Some compact files keep top-level papers empty but retain all first-stage
    # candidates in logs. Keep old score >= 2 answers for strict filtering.
    best_by_arxiv_id: dict[str, dict[str, Any]] = {}
    for log in row.get("logs") or []:
        if not isinstance(log, dict):
            continue
        for paper in log.get("papers") or []:
            if not isinstance(paper, dict):
                continue
            arxiv_id = normalize_arxiv_id(str(paper.get("arxiv_id") or ""))
            score = paper.get("score", paper.get("relevance_level"))
            if not arxiv_id or not isinstance(score, (int, float)) or float(score) < 2:
                continue
            previous = best_by_arxiv_id.get(arxiv_id)
            if previous is not None and float(previous.get("original_score") or 0) >= float(score):
                continue
            best_by_arxiv_id[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "original_score": score,
                "original_confidence": str(paper.get("confidence") or "").strip().lower(),
            }
    extracted = sorted(
        best_by_arxiv_id.values(),
        key=lambda item: (-float(item.get("original_score") or 0), str(item.get("arxiv_id") or "")),
    )
    return extracted


async def _hydrate_candidate(
    paper_client: PaperSearchV2Client,
    compact_paper: dict[str, Any],
    *,
    paper_index: int,
    max_abstract_chars: int,
) -> PaperCandidate | None:
    arxiv_id = str(compact_paper["arxiv_id"])
    paper = await paper_client.get_paper(arxiv_id)
    if paper is None or not paper.title or not paper.abstract:
        return None
    abstract = paper.abstract
    if max_abstract_chars > 0 and len(abstract) > max_abstract_chars:
        abstract = abstract[:max_abstract_chars].rstrip()
    return PaperCandidate(
        paper_index=paper_index,
        arxiv_id=arxiv_id,
        title=paper.title,
        abstract=abstract,
        original_score=compact_paper.get("original_score"),
        original_confidence=str(compact_paper.get("original_confidence") or ""),
    )


def _candidate_for_prompt(candidate: PaperCandidate) -> dict[str, Any]:
    return {
        "paper_index": candidate.paper_index,
        "arxiv_id": candidate.arxiv_id,
        "title": candidate.title,
        "abstract": candidate.abstract,
    }


def _chunks(items: list[PaperCandidate], size: int) -> list[list[PaperCandidate]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _parse_strict_score(value: Any) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or value not in {0, 1, 2}:
        raise ValueError(f"Invalid strict_score: {value!r}")
    return value


def _parse_confidence(value: Any) -> str:
    confidence = str(value or "").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid confidence: {value!r}")
    return confidence


def _parse_batch_response(raw_content: str, expected: list[PaperCandidate]) -> dict[int, dict[str, Any]]:
    payload = json.loads(_extract_json_object(raw_content))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Model response must contain a results list.")

    expected_by_index = {candidate.paper_index: candidate for candidate in expected}
    parsed: dict[int, dict[str, Any]] = {}
    for item in payload["results"]:
        if not isinstance(item, dict):
            raise ValueError("Each result must be a JSON object.")

        paper_index = item.get("paper_index")
        if isinstance(paper_index, str) and paper_index.strip().isdigit():
            paper_index = int(paper_index.strip())
        if not isinstance(paper_index, int) or paper_index not in expected_by_index:
            raise ValueError(f"Unexpected paper_index: {paper_index!r}")

        expected_candidate = expected_by_index[paper_index]
        arxiv_id = normalize_arxiv_id(str(item.get("arxiv_id") or ""))
        if arxiv_id and arxiv_id != expected_candidate.arxiv_id:
            raise ValueError(
                f"ArXiv ID mismatch for paper_index={paper_index}: "
                f"{arxiv_id!r} != {expected_candidate.arxiv_id!r}"
            )

        parsed[paper_index] = {
            "reason": str(item.get("reason") or "").strip(),
            "strict_score": _parse_strict_score(item.get("strict_score")),
            "confidence": _parse_confidence(item.get("confidence")),
        }

    missing_indexes = sorted(set(expected_by_index) - set(parsed))
    if missing_indexes:
        raise ValueError(f"Model response missed paper indexes: {missing_indexes}")
    return parsed


async def _call_model_batch_once(
    config: ModelConfig,
    *,
    query: str,
    batch: list[PaperCandidate],
    model_call_stats: dict[str, int],
) -> dict[int, dict[str, Any]]:
    if not config.api_key:
        raise ValueError(f"Missing API key for {config.label}. Fill {config.label} settings in AnswerFilters/.env.")

    user_prompt = build_strict_batch_user_prompt(
        query=query,
        papers=[_candidate_for_prompt(candidate) for candidate in batch],
    )
    model_call_stats[config.label] = model_call_stats.get(config.label, 0) + 1
    response = await asyncio.to_thread(
        call_openai_chat,
        model_name=config.model,
        system_prompt=STRICT_BATCH_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        api_key=config.api_key,
        api_base=config.api_base,
    )
    message_content = response.choices[0].message.content or ""
    return _parse_batch_response(message_content, batch)


async def _call_model_batch_with_retries(
    config: ModelConfig,
    *,
    query: str,
    batch: list[PaperCandidate],
    max_retries: int,
    model_call_stats: dict[str, int],
) -> dict[int, dict[str, Any]]:
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await _call_model_batch_once(
                config,
                query=query,
                batch=batch,
                model_call_stats=model_call_stats,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unreachable model retry state.")


def _failed_judgment(exc: BaseException) -> dict[str, Any]:
    return {
        "reason": "",
        "strict_score": None,
        "confidence": "",
        "error": str(exc),
    }


async def _score_batch_with_fallback(
    config: ModelConfig,
    *,
    query: str,
    batch: list[PaperCandidate],
    max_retries: int,
    model_call_stats: dict[str, int],
) -> dict[int, dict[str, Any]]:
    try:
        return await _call_model_batch_with_retries(
            config,
            query=query,
            batch=batch,
            max_retries=max_retries,
            model_call_stats=model_call_stats,
        )
    except Exception as batch_exc:
        if len(batch) == 1:
            return {batch[0].paper_index: _failed_judgment(batch_exc)}

    results: dict[int, dict[str, Any]] = {}
    for candidate in batch:
        try:
            single_result = await _call_model_batch_with_retries(
                config,
                query=query,
                batch=[candidate],
                max_retries=max_retries,
                model_call_stats=model_call_stats,
            )
            results.update(single_result)
        except Exception as exc:
            results[candidate.paper_index] = _failed_judgment(exc)
    return results


async def _score_candidates(
    config: ModelConfig,
    *,
    query: str,
    candidates: list[PaperCandidate],
    batch_size: int,
    max_retries: int,
    model_call_stats: dict[str, int],
    request_concurrency: int,
) -> dict[int, dict[str, Any]]:
    all_results: dict[int, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(max(1, request_concurrency))

    async def score_one_batch(batch: list[PaperCandidate]) -> dict[int, dict[str, Any]]:
        async with semaphore:
            return await _score_batch_with_fallback(
                config,
                query=query,
                batch=batch,
                max_retries=max_retries,
                model_call_stats=model_call_stats,
            )

    tasks = [asyncio.create_task(score_one_batch(batch)) for batch in _chunks(candidates, batch_size)]
    for batch_results in await asyncio.gather(*tasks):
        all_results.update(batch_results)
    return all_results


def _is_strict_pass(judgment: dict[str, Any] | None) -> bool:
    return judgment is not None and judgment.get("strict_score") == 2


def _short_reason(judgment: dict[str, Any] | None, *, max_chars: int = 240) -> str:
    if judgment is None:
        return ""
    reason = " ".join(str(judgment.get("reason") or "").split())
    if len(reason) <= max_chars:
        return reason
    return reason[:max_chars].rstrip()


def _write_summary(
    *,
    summary_path: Path,
    input_path: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    start_time: float,
) -> None:
    completed = [row for row in records if row.get("status") == "ok"]
    summary = {
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "processed_count": len(records) + len(errors),
        "completed_count": len(completed),
        "error_count": len(errors),
        "total_candidate_count": sum(int(row.get("candidate_count") or 0) for row in completed),
        "total_hydrated_count": sum(int(row.get("hydrated_count") or 0) for row in completed),
        "total_qwen_model_call_count": sum(int(row.get("qwen_model_call_count") or 0) for row in completed),
        "total_deepseek_model_call_count": sum(int(row.get("deepseek_model_call_count") or 0) for row in completed),
        "total_final_answer_count": sum(int(row.get("final_answer_count") or 0) for row in completed),
        "total_final_rejected_count": sum(int(row.get("final_rejected_count") or 0) for row in completed),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "records": records,
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


async def _process_row(
    row: dict[str, Any],
    *,
    paper_client: PaperSearchV2Client,
    qwen_config: ModelConfig,
    deepseek_config: ModelConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_time = time.monotonic()
    query_id = str(row.get("query_id") or "").strip()
    query = str(row.get("final_query") or "").strip()
    if not query_id:
        raise ValueError("Input row is missing query_id.")
    if not query:
        raise ValueError(f"Input row is missing final_query: {query_id}")

    compact_papers = _extract_compact_papers(row)
    hydrated_results = await asyncio.gather(
        *(
            _hydrate_candidate(
                paper_client,
                compact_paper,
                paper_index=index,
                max_abstract_chars=args.max_abstract_chars,
            )
            for index, compact_paper in enumerate(compact_papers, start=1)
        ),
        return_exceptions=True,
    )
    candidates: list[PaperCandidate] = []
    missing_paper_ids: list[str] = []
    for compact_paper, hydrated in zip(compact_papers, hydrated_results):
        if isinstance(hydrated, PaperCandidate):
            candidates.append(hydrated)
        else:
            missing_paper_ids.append(str(compact_paper.get("arxiv_id") or ""))

    model_call_stats = {"qwen": 0, "deepseek": 0}
    qwen_results, deepseek_results = await asyncio.gather(
        _score_candidates(
            qwen_config,
            query=query,
            candidates=candidates,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            model_call_stats=model_call_stats,
            request_concurrency=args.model_request_concurrency,
        ),
        _score_candidates(
            deepseek_config,
            query=query,
            candidates=candidates,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            model_call_stats=model_call_stats,
            request_concurrency=args.model_request_concurrency,
        ),
    )

    paper_results: list[dict[str, Any]] = []
    final_answer_arxiv_ids: list[str] = []
    failed_candidate_count = 0
    qwen_strict_2_count = 0
    deepseek_strict_2_count = 0
    for candidate in candidates:
        qwen_judgment = qwen_results.get(candidate.paper_index)
        deepseek_judgment = deepseek_results.get(candidate.paper_index)
        qwen_pass = _is_strict_pass(qwen_judgment)
        deepseek_pass = _is_strict_pass(deepseek_judgment)
        passed = qwen_pass and deepseek_pass
        if qwen_pass:
            qwen_strict_2_count += 1
        if deepseek_pass:
            deepseek_strict_2_count += 1
        if passed:
            final_answer_arxiv_ids.append(candidate.arxiv_id)
        if qwen_judgment is None or deepseek_judgment is None or "error" in qwen_judgment or "error" in deepseek_judgment:
            failed_candidate_count += 1

        final_score = 2 if passed else 0
        final_status = "accepted" if passed else "rejected"
        paper_results.append(
            {
                "paper_index": candidate.paper_index,
                "arxiv_id": candidate.arxiv_id,
                "title": candidate.title,
                "previous_stage_score": candidate.original_score,
                "previous_stage_confidence": candidate.original_confidence,
                "original_score": candidate.original_score,
                "original_confidence": candidate.original_confidence,
                "qwen_strict_score": None if qwen_judgment is None else qwen_judgment.get("strict_score"),
                "qwen_confidence": None if qwen_judgment is None else qwen_judgment.get("confidence"),
                "qwen_reason_brief": _short_reason(qwen_judgment),
                "qwen_short_reason": _short_reason(qwen_judgment),
                "deepseek_strict_score": None if deepseek_judgment is None else deepseek_judgment.get("strict_score"),
                "deepseek_confidence": None if deepseek_judgment is None else deepseek_judgment.get("confidence"),
                "deepseek_reason_brief": _short_reason(deepseek_judgment),
                "deepseek_short_reason": _short_reason(deepseek_judgment),
                "final_strict_score": final_score,
                "final_score": final_score,
                "final_status": final_status,
                "qwen": qwen_judgment,
                "deepseek": deepseek_judgment,
                "passed": passed,
            }
        )

    elapsed_seconds = round(time.monotonic() - start_time, 3)
    final_answer_count = len(final_answer_arxiv_ids)
    final_rejected_count = len(candidates) - final_answer_count
    count_row = {
        "query_id": query_id,
        "final_query": query,
        "candidate_count": len(compact_papers),
        "hydrated_count": len(candidates),
        "missing_paper_count": len(missing_paper_ids),
        "qwen_model_call_count": model_call_stats["qwen"],
        "deepseek_model_call_count": model_call_stats["deepseek"],
        "qwen_strict_2_count": qwen_strict_2_count,
        "deepseek_strict_2_count": deepseek_strict_2_count,
        "final_answer_count": final_answer_count,
        "final_accepted_count": final_answer_count,
        "final_rejected_count": final_rejected_count,
        "failed_candidate_count": failed_candidate_count,
        "batch_size": args.batch_size,
        "status": "ok",
        "elapsed_seconds": elapsed_seconds,
    }
    detail_row = {
        **count_row,
        "input_record": {
            "query_id": query_id,
            "final_query": query,
            "answer_count": row.get("answer_count"),
            "extra_weak_answer_count": row.get("extra_weak_answer_count"),
            "weakly_relevant_count": row.get("weakly_relevant_count"),
        },
        "missing_paper_ids": missing_paper_ids,
        "final_answer_arxiv_ids": final_answer_arxiv_ids,
        "papers": paper_results,
    }
    return count_row, detail_row


async def main_async() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.query_concurrency < 1:
        raise ValueError("--query-concurrency must be at least 1.")
    if args.model_request_concurrency < 1:
        raise ValueError("--model-request-concurrency must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")

    qwen_config, deepseek_config = _model_configs()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    counts_path = output_dir / "strict_filter_counts.jsonl"
    details_path = output_dir / "strict_filter_details.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "strict_filter_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _read_jsonl(input_path)
    selected_rows = _slice_rows(all_rows, start=args.start, limit=args.limit)
    skipped_ids = set() if args.no_resume else _completed_query_ids(counts_path, details_path)
    existing_records = _read_jsonl(counts_path)
    existing_errors = _read_jsonl(errors_path)
    run_records: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []
    write_lock = asyncio.Lock()
    query_semaphore = asyncio.Semaphore(args.query_concurrency)
    paper_client = PaperSearchV2Client(timeout=30.0)
    start_time = time.monotonic()

    async def process_one(index: int, row: dict[str, Any]) -> None:
        query_id = str(row.get("query_id") or "").strip()
        if query_id in skipped_ids:
            print(f"Skipping completed query {query_id}", flush=True)
            return

        async with query_semaphore:
            print(f"Processing {index}/{len(selected_rows)}: {query_id}", flush=True)
            try:
                count_row, detail_row = await _process_row(
                    row,
                    paper_client=paper_client,
                    qwen_config=qwen_config,
                    deepseek_config=deepseek_config,
                    args=args,
                )
                async with write_lock:
                    _append_jsonl(counts_path, count_row)
                    _append_jsonl(details_path, detail_row)
                    run_records.append(count_row)
                    _write_summary(
                        summary_path=summary_path,
                        input_path=input_path,
                        output_dir=output_dir,
                        records=existing_records + run_records,
                        errors=existing_errors + run_errors,
                        start_time=start_time,
                    )
                print(f"Completed {query_id}: {count_row['final_answer_count']} final answers", flush=True)
            except Exception as exc:
                error_row = {
                    "query_id": query_id,
                    "final_query": str(row.get("final_query") or ""),
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                count_error_row = {key: value for key, value in error_row.items() if key != "traceback"}
                async with write_lock:
                    _append_jsonl(counts_path, count_error_row)
                    _append_jsonl(errors_path, error_row)
                    run_errors.append(error_row)
                    _write_summary(
                        summary_path=summary_path,
                        input_path=input_path,
                        output_dir=output_dir,
                        records=existing_records + run_records,
                        errors=existing_errors + run_errors,
                        start_time=start_time,
                    )
                print(f"Failed {query_id}: {exc}", flush=True)

    try:
        tasks = [
            asyncio.create_task(process_one(index, row))
            for index, row in enumerate(selected_rows, start=1)
        ]
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        await paper_client.close()

    _write_summary(
        summary_path=summary_path,
        input_path=input_path,
        output_dir=output_dir,
        records=existing_records + run_records,
        errors=existing_errors + run_errors,
        start_time=start_time,
    )
    print(f"Summary: {summary_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
