from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paperbench.enums import QueryType
from paperbench.io_utils import write_jsonl
from paperbench.types import TopicSeed


METHOD_KEYWORDS = (
    "algorithm",
    "algorithms",
    "architecture",
    "architectures",
    "bayesian",
    "graph neural",
    "knowledge graph",
    "logic programming",
    "markov decision process",
    "model",
    "models",
    "network",
    "networks",
    "ontology",
    "ontologies",
    "planning",
    "rdf",
    "sat solving",
    "search",
    "theorem proving",
    "transformer",
    "transformers",
)

COLLECTION_KEYWORDS = (
    "benchmark",
    "benchmarks",
    "corpora",
    "corpus",
    "dataset",
    "datasets",
    "repository",
    "repositories",
    "survey",
    "surveys",
    "taxonomy",
    "taxonomies",
)

TYPE_SUFFIX = {
    QueryType.SPECIFIC_TOPIC: "specific",
    QueryType.CROSS_TOPIC: "cross",
    QueryType.METHOD_ARCHITECTURE: "method",
    QueryType.COLLECTION_SCOPING: "collection",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert stratified raw topic samples into standard TopicSeed records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "topic_seeds_stratified.jsonl",
        help="Input JSONL with stratified topic samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "topic_seeds_for_query_generation.jsonl",
        help="Output JSONL path for TopicSeed records.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield payload


def normalize_label(label: str) -> str:
    return " ".join(label.split())


def aggregate_samples(path: Path) -> tuple[Counter[tuple[str, str, str]], int]:
    counts: Counter[tuple[str, str, str]] = Counter()
    total_rows = 0
    for row in iter_jsonl(path):
        stratum = str(row.get("stratum", "")).strip()
        acm_id = str(row.get("acm_id", "")).strip()
        leaf_label = normalize_label(str(row.get("leaf_label", "")).strip())
        if not stratum or not acm_id or not leaf_label:
            continue
        counts[(stratum, acm_id, leaf_label)] += 1
        total_rows += 1
    return counts, total_rows


def looks_cross_topic(label: str) -> bool:
    return " for " in label or " with " in label or " and " in label


def assign_query_types(leaf_label: str) -> list[QueryType]:
    normalized = leaf_label.lower()
    assigned: list[QueryType] = [QueryType.SPECIFIC_TOPIC]

    if any(keyword in normalized for keyword in METHOD_KEYWORDS):
        assigned.append(QueryType.METHOD_ARCHITECTURE)

    if any(keyword in normalized for keyword in COLLECTION_KEYWORDS):
        assigned.append(QueryType.COLLECTION_SCOPING)

    if looks_cross_topic(normalized):
        assigned.append(QueryType.CROSS_TOPIC)

    deduped: list[QueryType] = []
    for query_type in assigned:
        if query_type not in deduped:
            deduped.append(query_type)
    return deduped[:3]


def difficulty_for_type(query_type: QueryType) -> str:
    if query_type == QueryType.SPECIFIC_TOPIC:
        return "easy"
    return "medium"


def build_topic_seeds(counts: Counter[tuple[str, str, str]]) -> list[TopicSeed]:
    seeds: list[TopicSeed] = []
    ordered_items = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][2].lower(), item[0][1]),
    )

    next_index = 1
    for (stratum, acm_id, leaf_label), sample_count in ordered_items:
        for query_type in assign_query_types(leaf_label):
            seeds.append(
                TopicSeed(
                    topic_id=f"TS_STRAT_{next_index:06d}_{TYPE_SUFFIX[query_type]}",
                    topic=leaf_label,
                    domain=stratum,
                    target_query_type=query_type,
                    difficulty_hint=difficulty_for_type(query_type),
                    sample_count=sample_count,
                    source_stratum=stratum,
                    source_acm_id=acm_id,
                    source_leaf_label=leaf_label,
                )
            )
            next_index += 1
    return seeds


def build_report(
    *,
    total_rows: int,
    unique_topics: int,
    seeds: list[TopicSeed],
) -> dict[str, object]:
    by_type: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    sample_count_histogram: Counter[int] = Counter()

    for seed in seeds:
        query_type = seed.target_query_type.value if seed.target_query_type is not None else "unknown"
        by_type[query_type] += 1
        by_domain[seed.domain] += 1
        if seed.sample_count is not None:
            sample_count_histogram[seed.sample_count] += 1

    return {
        "total_input_rows": total_rows,
        "unique_topics": unique_topics,
        "output_topic_seeds": len(seeds),
        "query_type_counts": dict(sorted(by_type.items())),
        "domain_counts": dict(sorted(by_domain.items())),
        "sample_count_histogram": {
            str(sample_count): count
            for sample_count, count in sorted(sample_count_histogram.items())
        },
    }


def main() -> None:
    args = parse_args()
    counts, total_rows = aggregate_samples(args.input)
    seeds = build_topic_seeds(counts)

    write_jsonl(args.output, seeds)

    report = build_report(
        total_rows=total_rows,
        unique_topics=len(counts),
        seeds=seeds,
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Input rows: {total_rows}")
    print(f"Unique topics: {len(counts)}")
    print(f"Output TopicSeed records: {len(seeds)}")
    for query_type, count in sorted(report["query_type_counts"].items()):
        print(f"  {query_type}: {count}")


if __name__ == "__main__":
    main()
