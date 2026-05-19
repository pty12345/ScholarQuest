from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from openai import OpenAI


ANSWER_FILTER_ROOT = Path(__file__).resolve().parent
ROOT = ANSWER_FILTER_ROOT.parents[2]
DEFAULT_INPUT_DIR = ANSWER_FILTER_ROOT / "output" / "qwen_batch_inputs_first347"
DEFAULT_RESULTS_PATH = DEFAULT_INPUT_DIR / "qwen_batch_submission_results.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_INPUT_DIR / "qwen_batch_submission_summary.json"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENVS = (
    "ANSWER_FILTER_QWEN_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Qwen-Max strict-filter batch request files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with qwen_batch_requests_*.jsonl.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="JSONL path for submission records.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="JSON path for submission summary.")
    parser.add_argument("--api-key-env", default="", help="Optional explicit API key environment variable.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible DashScope base URL.")
    parser.add_argument("--completion-window", default="24h", choices=("24h",), help="Batch completion window.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of request files to submit.")
    parser.add_argument("--start-index", type=int, default=1, help="One-based index of the first sorted request file.")
    parser.add_argument("--end-index", type=int, default=None, help="One-based index of the last sorted request file.")
    return parser.parse_args()


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ANSWER_FILTER_ROOT / ".env")


def _resolve_api_key(api_key_env: str) -> tuple[str, str]:
    env_names = (api_key_env,) if api_key_env else DEFAULT_API_KEY_ENVS
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value, env_name
    raise ValueError(f"Missing API key. Set one of: {', '.join(DEFAULT_API_KEY_ENVS)}.")


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
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(payload)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _request_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _submit_one_with_retry(
    *,
    client: OpenAI,
    input_path: Path,
    completion_window: str,
    max_attempts: int = 3,
) -> tuple[Any, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            uploaded_file = client.files.create(file=input_path, purpose="batch")
            batch = client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window=cast(Literal["24h"], completion_window),
                metadata={"ds_name": input_path.stem},
            )
            return uploaded_file, batch
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            print(f"Retry {input_path.name}: attempt {attempt}/{max_attempts} failed: {exc}", flush=True)
            time.sleep(5 * attempt)
    assert last_error is not None
    raise last_error


def _select_input_paths(input_dir: Path, *, start_index: int, end_index: int | None, limit: int | None) -> list[Path]:
    if start_index <= 0:
        raise ValueError("--start-index must be positive.")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    input_paths = sorted(input_dir.glob("qwen_batch_requests_*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No qwen_batch_requests_*.jsonl files found in {input_dir}")
    selected = input_paths[start_index - 1 : end_index]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _write_summary(
    *,
    input_dir: Path,
    results_path: Path,
    summary_path: Path,
    rows: list[dict[str, Any]],
    api_base: str,
    model: str,
) -> None:
    submitted_rows = [row for row in rows if row.get("status") == "submitted"]
    summary = {
        "input_dir": str(input_dir.resolve()),
        "results_path": str(results_path.resolve()),
        "submitted_count": len(submitted_rows),
        "submitted_request_count": sum(int(row.get("request_count") or 0) for row in submitted_rows),
        "batch_ids": [row.get("batch_id") for row in submitted_rows if row.get("batch_id")],
        "api_base": api_base,
        "model": model,
        "generated_at_unix_time": int(time.time()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _load_env()
    api_key, api_key_env = _resolve_api_key(args.api_key_env)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    input_dir = args.input_dir.resolve()
    results_path = args.results_path.resolve()
    summary_path = args.summary_path.resolve()
    input_paths = _select_input_paths(
        input_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )

    existing_rows = _read_jsonl(results_path)
    submitted_paths = {str(row.get("request_file") or "") for row in existing_rows if row.get("status") == "submitted"}
    all_rows = list(existing_rows)
    model = "qwen-max"

    for input_path in input_paths:
        if str(input_path) in submitted_paths:
            print(f"Skip already submitted file: {input_path.name}", flush=True)
            continue

        request_count = _request_count(input_path)
        try:
            uploaded_file, batch = _submit_one_with_retry(
                client=client,
                input_path=input_path,
                completion_window=args.completion_window,
            )
        except Exception as exc:
            row = {
                "status": "error",
                "request_file": str(input_path),
                "request_count": request_count,
                "error": f"{type(exc).__name__}: {exc}",
                "api_base": args.base_url,
                "api_key_env": api_key_env,
                "model": model,
                "submitted_unix_time": int(time.time()),
            }
            _append_jsonl(results_path, row)
            all_rows.append(row)
            print(f"Failed {input_path.name}: {row['error']}", flush=True)
            continue
        row = {
            "status": "submitted",
            "request_file": str(input_path),
            "request_count": request_count,
            "uploaded_file_id": uploaded_file.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
            "created_at": getattr(batch, "created_at", None),
            "endpoint": "/v1/chat/completions",
            "completion_window": args.completion_window,
            "api_base": args.base_url,
            "api_key_env": api_key_env,
            "model": model,
            "submitted_unix_time": int(time.time()),
        }
        _append_jsonl(results_path, row)
        all_rows.append(row)
        print(f"Submitted {input_path.name}: batch_id={batch.id}, status={batch.status}", flush=True)

    _write_summary(
        input_dir=input_dir,
        results_path=results_path,
        summary_path=summary_path,
        rows=all_rows,
        api_base=args.base_url,
        model=model,
    )
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
