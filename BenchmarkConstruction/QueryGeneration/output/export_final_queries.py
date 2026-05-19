from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "pasa_query_records_sample10_v2.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "final_queries_sample10_v2.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final_query values from a JSONL file into a TXT file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output TXT file.",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "category", "grouped"),
        default="grouped",
        help="Export format for human review.",
    )
    return parser.parse_args()


def extract_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at line {line_no}.")

            final_query = payload.get("final_query")
            if not isinstance(final_query, str):
                raise ValueError(f"Missing or invalid final_query at line {line_no}.")

            records.append(payload)

    return records


def _read_topic_seed(payload: dict[str, Any]) -> str:
    topic_seed = payload.get("topic_seed")
    if isinstance(topic_seed, dict):
        topic = topic_seed.get("topic")
        if isinstance(topic, str):
            return topic
    return "unknown_topic"


def _read_category(payload: dict[str, Any]) -> str:
    category = payload.get("category")
    if isinstance(category, str):
        return category
    return "unknown_category"


def _format_plain(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for payload in records:
        lines.append(str(payload["final_query"]))
        lines.append("")
    return lines


def _format_category(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for payload in records:
        lines.append(f"[{_read_category(payload)}] {payload['final_query']}")
        lines.append("")
    return lines


def _format_grouped(records: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    ordered_topics: list[str] = []
    for payload in records:
        topic = _read_topic_seed(payload)
        if topic not in grouped:
            grouped[topic] = []
            ordered_topics.append(topic)
        grouped[topic].append(payload)

    lines: list[str] = []
    for topic in ordered_topics:
        lines.append(f"Topic seed: {topic}")
        for payload in grouped[topic]:
            lines.append(f"- [{_read_category(payload)}] {payload['final_query']}")
        lines.append("")
    return lines


def render_output(records: list[dict[str, Any]], export_format: str) -> list[str]:
    if export_format == "plain":
        return _format_plain(records)
    if export_format == "category":
        return _format_category(records)
    return _format_grouped(records)


def write_queries(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for line in lines:
            outfile.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    records = extract_records(args.input)
    lines = render_output(records, args.format)
    write_queries(args.output, lines)
    print(f"Exported {len(records)} final_query values to {args.output}")


if __name__ == "__main__":
    main()
