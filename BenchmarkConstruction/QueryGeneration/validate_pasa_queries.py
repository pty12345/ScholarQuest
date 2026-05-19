from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QUERY_GENERATION_DIR = Path(__file__).resolve().parent
if str(QUERY_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUERY_GENERATION_DIR))

from paperbench.io_utils import ensure_parent
from pasa_types import PasaCategory, PasaGenerationConfig, PasaQueryRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated PASA query records.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=QUERY_GENERATION_DIR / "pasa_generation_config.yaml",
        help="Path to the PASA generation config file.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional override for the query record JSONL file.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional override for the validation report path.",
    )
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, object], payload)


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _normalize_query(query: str) -> str:
    normalized = " ".join(query.split()).lower()
    return normalized.rstrip(".?!")


def _opening_signature(query: str, max_words: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z]+", query.lower())
    if not tokens:
        return ""
    return " ".join(tokens[:max_words])


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _parse_query_record(line: str) -> PasaQueryRecord:
    validator = getattr(PasaQueryRecord, "model_validate_json", None)
    if callable(validator):
        return cast(PasaQueryRecord, validator(line))
    parser = getattr(PasaQueryRecord, "parse_raw", None)
    if callable(parser):
        return cast(PasaQueryRecord, parser(line))
    raise TypeError("PasaQueryRecord does not support JSON parsing.")


def _read_query_records(path: Path) -> list[PasaQueryRecord]:
    if not path.exists():
        return []
    records: list[PasaQueryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            records.append(_parse_query_record(raw_line))
    return records


def main() -> None:
    args = parse_args()
    root_dir = args.root.resolve()
    config = PasaGenerationConfig(**_load_yaml_mapping(args.config.resolve()))
    input_path = (
        args.input_file.resolve()
        if args.input_file is not None
        else _resolve_path(root_dir, config.paths.output_file)
    )
    report_path = (
        args.report_file.resolve()
        if args.report_file is not None
        else _resolve_path(root_dir, "QueryGeneration/output/pasa_validation_report.json")
    )

    records = _read_query_records(input_path)
    required_categories = {category.value for category in PasaCategory}

    category_counts: Counter[str] = Counter()
    seed_to_categories: dict[str, set[str]] = defaultdict(set)
    seed_to_queries: dict[str, set[str]] = defaultdict(set)
    duplicate_query_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    missing_category_seeds: list[str] = []
    intra_seed_duplicate_seeds: list[str] = []

    for record in records:
        category_value = record.category.value
        category_counts[category_value] += 1
        seed_id = record.topic_seed.seed_id
        seed_to_categories[seed_id].add(category_value)
        normalized_query = _normalize_query(record.final_query)
        if normalized_query in seed_to_queries[seed_id]:
            intra_seed_duplicate_seeds.append(seed_id)
        seed_to_queries[seed_id].add(normalized_query)
        duplicate_query_counts[normalized_query] += 1
        opening = _opening_signature(record.final_query)
        if opening:
            opening_counts[opening] += 1

    for seed_id, categories in seed_to_categories.items():
        if categories != required_categories:
            missing_category_seeds.append(seed_id)

    global_duplicate_queries = sum(1 for count in duplicate_query_counts.values() if count > 1)
    unique_intra_seed_duplicate_seeds = sorted(set(intra_seed_duplicate_seeds))
    missing_category_seeds = sorted(missing_category_seeds)

    report = {
        "input_file": str(input_path),
        "record_count": len(records),
        "seed_count": len(seed_to_categories),
        "category_counts": _sorted_counter(category_counts),
        "complete_seed_count": sum(
            1 for categories in seed_to_categories.values() if categories == required_categories
        ),
        "missing_category_seed_count": len(missing_category_seeds),
        "missing_category_seed_examples": missing_category_seeds[:20],
        "intra_seed_duplicate_seed_count": len(unique_intra_seed_duplicate_seeds),
        "intra_seed_duplicate_seed_examples": unique_intra_seed_duplicate_seeds[:20],
        "global_exact_duplicate_query_count": global_duplicate_queries,
        "opening_counts_top10": dict(opening_counts.most_common(10)),
    }

    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validated records: {len(records)}")
    print(f"Validated seeds: {len(seed_to_categories)}")
    print(f"Seeds with all four categories: {report['complete_seed_count']}")
    print(f"Seeds missing categories: {report['missing_category_seed_count']}")
    print(f"Seeds with intra-seed duplicates: {report['intra_seed_duplicate_seed_count']}")
    print(f"Global exact duplicate queries: {report['global_exact_duplicate_query_count']}")
    print(f"Most common openings: {report['opening_counts_top10']}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
