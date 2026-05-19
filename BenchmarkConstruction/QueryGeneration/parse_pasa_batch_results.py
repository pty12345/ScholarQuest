from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "QueryGeneration" / "output" / "pasa_batch_results"
DEFAULT_OUTPUT_JSONL = ROOT / "QueryGeneration" / "output" / "pasa_batch_results_flat.jsonl"
DEFAULT_OUTPUT_TXT = ROOT / "QueryGeneration" / "output" / "pasa_batch_final_queries.txt"
DEFAULT_REPORT = ROOT / "QueryGeneration" / "output" / "pasa_batch_results_report.json"
EXPECTED_CATEGORIES = {
    "method_capability",
    "setting_anchor",
    "claim_comparison",
    "scope_control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten DashScope batch outputs into query records.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Directory with batch output JSONL files.")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Flattened JSONL output path.")
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT, help="Plain text query output path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Summary report JSON path.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _message_content(row: dict[str, Any]) -> tuple[str, str, int | None, int | None, int | None]:
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError("Missing response object.")
    if response.get("status_code") != 200:
        raise ValueError(f"Unexpected status_code: {response.get('status_code')}")

    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("Missing response body.")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid choice object.")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing message object.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Missing message content.")

    usage = body.get("usage")
    prompt_tokens = completion_tokens = total_tokens = None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None

    model = body.get("model")
    model_name = model if isinstance(model, str) else ""
    return content, model_name, prompt_tokens, completion_tokens, total_tokens


def _normalize_risk_flags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _flatten_success_row(
    row: dict[str, Any],
    *,
    source_file: Path,
    next_query_index: int,
) -> tuple[list[dict[str, Any]], int]:
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str):
        raise ValueError("Missing custom_id.")

    content, model_name, prompt_tokens, completion_tokens, total_tokens = _message_content(row)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Model content is not a JSON object.")
    queries = parsed.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Model content does not contain a queries list.")

    seed_id = parsed.get("seed_id")
    topic_seed = parsed.get("topic_seed")
    domain = parsed.get("domain")
    seed_id_text = seed_id if isinstance(seed_id, str) else custom_id.removeprefix("query-")
    topic_seed_text = topic_seed if isinstance(topic_seed, str) else ""
    domain_text = domain if isinstance(domain, str) else ""

    records: list[dict[str, Any]] = []
    query_index = next_query_index
    seen_categories: set[str] = set()
    for query_payload in queries:
        if not isinstance(query_payload, dict):
            continue
        category = query_payload.get("category")
        query = query_payload.get("query")
        if not isinstance(category, str) or not isinstance(query, str):
            continue
        seen_categories.add(category)
        records.append(
            {
                "query_id": f"BQ_{query_index:06d}",
                "custom_id": custom_id,
                "seed_id": seed_id_text,
                "topic_seed": topic_seed_text,
                "domain": domain_text,
                "category": category,
                "constraint_kind": query_payload.get("constraint_kind") if isinstance(query_payload.get("constraint_kind"), str) else "",
                "constraint_value": query_payload.get("constraint_value") if isinstance(query_payload.get("constraint_value"), str) else "",
                "final_query": query.strip(),
                "rationale": query_payload.get("rationale") if isinstance(query_payload.get("rationale"), str) else "",
                "risk_flags": _normalize_risk_flags(query_payload.get("risk_flags")),
                "llm_model": model_name,
                "source_batch_output": str(source_file),
                "request_id": row.get("id") if isinstance(row.get("id"), str) else "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        query_index += 1

    missing_categories = sorted(EXPECTED_CATEGORIES - seen_categories)
    if missing_categories:
        raise ValueError(f"Missing categories for {custom_id}: {missing_categories}")
    return records, query_index


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_files = sorted(result_dir.glob("*_output.jsonl"))
    error_files = sorted(result_dir.glob("*_error.jsonl"))
    if not output_files:
        raise FileNotFoundError(f"No output JSONL files found in {result_dir}")

    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    next_query_index = 1

    for output_file in output_files:
        for row in _read_jsonl(output_file):
            try:
                row_records, next_query_index = _flatten_success_row(
                    row,
                    source_file=output_file,
                    next_query_index=next_query_index,
                )
                records.extend(row_records)
                source_counts[output_file.name] += 1
                for record in row_records:
                    category_counts[str(record["category"])] += 1
            except Exception as exc:
                parse_errors.append(
                    {
                        "source_file": str(output_file),
                        "custom_id": str(row.get("custom_id", "")),
                        "error": str(exc),
                    }
                )

    _write_jsonl(args.output_jsonl.resolve(), records)
    args.output_txt.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.resolve().write_text(
        "\n".join(str(record["final_query"]) for record in records) + "\n",
        encoding="utf-8",
    )

    error_rows = sum(len(_read_jsonl(path)) for path in error_files)
    report = {
        "batch_output_files": [str(path) for path in output_files],
        "batch_error_files": [str(path) for path in error_files],
        "successful_response_count": sum(source_counts.values()),
        "flattened_query_count": len(records),
        "batch_error_row_count": error_rows,
        "parse_error_count": len(parse_errors),
        "category_counts": dict(sorted(category_counts.items())),
        "source_response_counts": dict(sorted(source_counts.items())),
        "parse_errors": parse_errors[:50],
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_txt": str(args.output_txt.resolve()),
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Successful responses: {report['successful_response_count']}")
    print(f"Flattened queries: {len(records)}")
    print(f"Batch error rows: {error_rows}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Output JSONL: {args.output_jsonl.resolve()}")
    print(f"Output TXT: {args.output_txt.resolve()}")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
