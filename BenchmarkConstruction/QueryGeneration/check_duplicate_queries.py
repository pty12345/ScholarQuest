from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "QueryGeneration" / "output" / "pasa_batch_results_flat.jsonl"
DEFAULT_REPORT = ROOT / "QueryGeneration" / "output" / "pasa_batch_duplicate_queries.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact duplicate query strings in a JSONL result file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file with final_query fields.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output duplicate report JSON file.")
    parser.add_argument(
        "--query-field",
        default="final_query",
        help="Field name that stores the query string.",
    )
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


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    rows = _read_jsonl(input_path)

    query_counter: Counter[str] = Counter()
    query_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_or_non_string_count = 0

    for row_index, row in enumerate(rows, start=1):
        value = row.get(args.query_field)
        if not isinstance(value, str):
            missing_or_non_string_count += 1
            continue

        # Exact string match: no lowercasing, punctuation changes, or whitespace normalization.
        query = value
        query_counter[query] += 1
        if query_counter[query] <= 20:
            query_rows[query].append(
                {
                    "row_index": row_index,
                    "query_id": row.get("query_id"),
                    "seed_id": row.get("seed_id"),
                    "topic_seed": row.get("topic_seed"),
                    "category": row.get("category"),
                }
            )

    duplicate_row_count = sum(count - 1 for count in query_counter.values() if count > 1)
    duplicate_queries = [
        {
            "query": query,
            "count": count,
            "examples": query_rows[query],
        }
        for query, count in query_counter.most_common()
        if count > 1
    ]

    report = {
        "input": str(input_path),
        "query_field": args.query_field,
        "total_rows": len(rows),
        "valid_query_count": sum(query_counter.values()),
        "unique_query_count": len(query_counter),
        "duplicate_query_string_count": len(duplicate_queries),
        "duplicate_row_count": duplicate_row_count,
        "missing_or_non_string_count": missing_or_non_string_count,
        "duplicates": duplicate_queries,
    }

    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Total rows: {report['total_rows']}")
    print(f"Valid queries: {report['valid_query_count']}")
    print(f"Unique query strings: {report['unique_query_count']}")
    print(f"Duplicate query strings: {report['duplicate_query_string_count']}")
    print(f"Duplicate rows beyond first occurrences: {report['duplicate_row_count']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
