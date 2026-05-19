from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _add_src_to_path  # noqa: F401

from paperbench.io_utils import read_jsonl
from paperbench.types import QueryRecord


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {key}: {value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect query distribution from a JSONL file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/adjudicated/query_pool_v0.1.jsonl"),
        help="Path to the query JSONL file.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input, QueryRecord)
    type_counter: Counter[str] = Counter()
    target_type_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    scope_counter: Counter[str] = Counter()
    adjudication_counter: Counter[str] = Counter()
    consistency_counter: Counter[str] = Counter()

    for record in records:
        type_counter[record.query_type.value if record.query_type else "unknown"] += 1
        target_type_counter[record.target_query_type.value if record.target_query_type else "unknown"] += 1
        difficulty_counter[record.difficulty.value if record.difficulty else "unknown"] += 1
        domain_counter[record.domain or "unknown"] += 1
        scope_counter[record.scope.value if record.scope else "unknown"] += 1
        adjudication_counter[record.adjudication_status.value if record.adjudication_status else "unknown"] += 1
        consistency_counter[str(record.type_consistent)] += 1

    print(f"Total records: {len(records)}")
    print()
    print_counter("Target query type distribution", target_type_counter)
    print_counter("Predicted query type distribution", type_counter)
    print_counter("Type consistency distribution", consistency_counter)
    print_counter("Query type distribution", type_counter)
    print_counter("Difficulty distribution", difficulty_counter)
    print_counter("Scope distribution", scope_counter)
    print_counter("Domain distribution", domain_counter)
    print_counter("Adjudication distribution", adjudication_counter)

    passing = sum(1 for r in records if r.quality_pass_count() >= 4)
    print(f"Quality pass rate (>=4/4 checks): {passing}/{len(records)}")


if __name__ == "__main__":
    main()
