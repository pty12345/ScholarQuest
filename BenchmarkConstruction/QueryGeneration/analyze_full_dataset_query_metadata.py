from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


QUERY_GENERATION_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = QUERY_GENERATION_DIR.parents[1]
DEFAULT_ANSWER_FILE = WORKSPACE_ROOT / "PaperBenchEval" / "dataset" / "FullDataset" / "final_answer_list_merged.jsonl"
DEFAULT_QUERY_METADATA_FILE = QUERY_GENERATION_DIR / "output" / "pasa_batch_unique_queries.jsonl"
DEFAULT_OUTPUT_DIR = QUERY_GENERATION_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join FullDataset answer counts with QueryGeneration metadata and summarize "
            "topic seeds and query categories for a selected answer-count range."
        )
    )
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWER_FILE, help="FullDataset answer JSONL.")
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=DEFAULT_QUERY_METADATA_FILE,
        help="QueryGeneration JSONL with query_id, seed_id, topic_seed, and category fields.",
    )
    parser.add_argument("--min-answers", type=int, default=5, help="Inclusive lower answer-count bound.")
    parser.add_argument("--max-answers", type=int, default=200, help="Inclusive upper answer-count bound.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument(
        "--output-prefix",
        default="full_dataset_answer_5_200_query_metadata",
        help="Prefix for summary JSON and joined CSV outputs.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def build_metadata_by_query_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata_by_query_id: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Missing query_id in metadata row {row_index}")
        if query_id in metadata_by_query_id:
            raise ValueError(f"Duplicate query_id in metadata: {query_id}")
        metadata_by_query_id[query_id] = row
    return metadata_by_query_id


def counter_to_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common()]


def write_joined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_id",
        "final_answer_count",
        "seed_id",
        "topic_seed",
        "domain",
        "category",
        "constraint_kind",
        "constraint_value",
        "final_query",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    if args.min_answers > args.max_answers:
        raise ValueError("--min-answers must be <= --max-answers")

    answer_rows = read_jsonl(args.answers.resolve())
    metadata_rows = read_jsonl(args.query_metadata.resolve())
    metadata_by_query_id = build_metadata_by_query_id(metadata_rows)

    selected_rows: list[dict[str, Any]] = []
    missing_metadata_query_ids: list[str] = []
    query_text_mismatches: list[dict[str, str]] = []

    for answer_row in answer_rows:
        query_id = str(answer_row["query_id"])
        answer_count = int(answer_row["final_answer_count"])
        if not args.min_answers <= answer_count <= args.max_answers:
            continue

        metadata = metadata_by_query_id.get(query_id)
        if metadata is None:
            missing_metadata_query_ids.append(query_id)
            continue

        answer_query = str(answer_row.get("final_query", ""))
        metadata_query = str(metadata.get("final_query", ""))
        if answer_query != metadata_query:
            query_text_mismatches.append(
                {
                    "query_id": query_id,
                    "answer_file_final_query": answer_query,
                    "metadata_final_query": metadata_query,
                }
            )

        selected_rows.append(
            {
                "query_id": query_id,
                "final_answer_count": answer_count,
                "seed_id": str(metadata.get("seed_id", "")),
                "topic_seed": str(metadata.get("topic_seed", "")),
                "domain": str(metadata.get("domain", "")),
                "category": str(metadata.get("category", "")),
                "constraint_kind": str(metadata.get("constraint_kind", "")),
                "constraint_value": str(metadata.get("constraint_value", "")),
                "final_query": answer_query,
            }
        )

    seed_id_counter = Counter(row["seed_id"] for row in selected_rows)
    topic_seed_counter = Counter(row["topic_seed"] for row in selected_rows)
    category_counter = Counter(row["category"] for row in selected_rows)
    domain_counter = Counter(row["domain"] for row in selected_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    csv_path = output_dir / f"{args.output_prefix}_joined.csv"

    summary = {
        "answers_path": str(args.answers.resolve()),
        "query_metadata_path": str(args.query_metadata.resolve()),
        "answer_count_range": {
            "min": args.min_answers,
            "max": args.max_answers,
            "inclusive": True,
        },
        "total_answer_rows": len(answer_rows),
        "total_metadata_rows": len(metadata_rows),
        "selected_query_count": len(selected_rows),
        "matched_query_count": len(selected_rows),
        "missing_metadata_count": len(missing_metadata_query_ids),
        "missing_metadata_query_ids": missing_metadata_query_ids,
        "query_text_mismatch_count": len(query_text_mismatches),
        "query_text_mismatches": query_text_mismatches[:50],
        "unique_seed_id_count": len(seed_id_counter),
        "unique_topic_seed_count": len(topic_seed_counter),
        "topic_seed_distribution": counter_to_rows(topic_seed_counter),
        "seed_id_distribution": counter_to_rows(seed_id_counter),
        "category_distribution": counter_to_rows(category_counter),
        "domain_distribution": counter_to_rows(domain_counter),
        "outputs": {
            "summary": str(summary_path),
            "joined_csv": str(csv_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_joined_csv(csv_path, selected_rows)

    print(f"Selected queries ({args.min_answers}-{args.max_answers} answers): {len(selected_rows)}")
    print(f"Unique seed_id count: {len(seed_id_counter)}")
    print(f"Unique topic_seed count: {len(topic_seed_counter)}")
    print("Category distribution:")
    for category, count in category_counter.most_common():
        print(f"  {category}: {count}")
    print(f"Summary: {summary_path}")
    print(f"Joined CSV: {csv_path}")


if __name__ == "__main__":
    main()
