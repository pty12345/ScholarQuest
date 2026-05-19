from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "QueryGeneration" / "output" / "pasa_batch_results_flat.jsonl"
DEFAULT_OUTPUT_JSONL = ROOT / "QueryGeneration" / "output" / "pasa_batch_unique_queries.jsonl"
DEFAULT_OUTPUT_TXT = ROOT / "QueryGeneration" / "output" / "pasa_batch_unique_queries.txt"
DEFAULT_REPORT = ROOT / "QueryGeneration" / "output" / "pasa_batch_unique_queries_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the first complete JSONL row for each exact query string.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file with final_query fields.")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL with unique queries.")
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT, help="Output text file with unique queries only.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Deduplication report JSON path.")
    parser.add_argument("--query-field", default="final_query", help="Field name that stores the query string.")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    rows = _iter_jsonl(input_path)

    seen_queries: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    missing_or_non_string_count = 0

    for row in rows:
        value = row.get(args.query_field)
        if not isinstance(value, str):
            missing_or_non_string_count += 1
            continue
        if value in seen_queries:
            duplicate_count += 1
            continue
        seen_queries.add(value)
        unique_rows.append(row)

    output_jsonl = args.output_jsonl.resolve()
    output_txt = args.output_txt.resolve()
    report_path = args.report.resolve()

    _write_jsonl(output_jsonl, unique_rows)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(
        "\n".join(str(row[args.query_field]) for row in unique_rows) + "\n",
        encoding="utf-8",
    )

    report = {
        "input": str(input_path),
        "query_field": args.query_field,
        "total_rows": len(rows),
        "unique_query_count": len(unique_rows),
        "removed_duplicate_count": duplicate_count,
        "missing_or_non_string_count": missing_or_non_string_count,
        "output_jsonl": str(output_jsonl),
        "output_txt": str(output_txt),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Total rows: {len(rows)}")
    print(f"Unique queries: {len(unique_rows)}")
    print(f"Removed duplicates: {duplicate_count}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Output TXT: {output_txt}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
