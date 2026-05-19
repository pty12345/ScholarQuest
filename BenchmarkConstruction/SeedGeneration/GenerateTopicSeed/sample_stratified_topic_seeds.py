from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Stratified topic seed sampling: arXiv cs.* weights from category_counts_all.csv, "
            "buckets from acm_ccs2012_leaf_to_arxiv_cs.jsonl matched_categories."
        )
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=here / "category_counts_all.csv",
        help="CSV with columns category,count (uses cs.* rows as stratum weights).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=here / "acm_ccs2012_leaf_to_arxiv_cs.jsonl",
        help="ACM leaf -> arXiv cs JSONL produced by TopicBucket classification.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Total number of topic seeds to draw.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for numpy Generator (reproducible allocation and within-bucket sampling).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path; each line is one sampled seed with metadata.",
    )
    parser.add_argument(
        "--allocation-report",
        type=Path,
        default=None,
        help="Optional JSON path: per-stratum counts, weights, and allocation.",
    )
    return parser.parse_args()


def load_cs_category_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        for row in reader:
            cat = row["category"].strip()
            if not cat.startswith("cs."):
                continue
            out[cat] = int(row["count"])
    if not out:
        raise ValueError(f"No cs.* rows found in {path}")
    return out


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from e


def build_cs_buckets(mapping_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Map cs.* -> list of (acm_id, leaf_label) for non-discarded leaves."""
    buckets: DefaultDict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_pair_per_bucket: DefaultDict[str, set[tuple[str, str]]] = defaultdict(set)

    for row in iter_jsonl(mapping_path):
        if row.get("discarded", True):
            continue
        acm_id = str(row.get("acm_id", ""))
        leaf = str(row.get("leaf_label", ""))
        matched = row.get("matched_categories") or []
        if not isinstance(matched, list):
            continue
        for raw in matched:
            cat = str(raw).strip()
            if not cat.startswith("cs."):
                continue
            key = (acm_id, leaf)
            if key in seen_pair_per_bucket[cat]:
                continue
            seen_pair_per_bucket[cat].add(key)
            buckets[cat].append(key)

    return {k: list(v) for k, v in buckets.items()}


def largest_remainder_allocation(n: int, weights: list[float]) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    exact = [n * w / s for w in weights]
    floors = [int(x) for x in exact]
    remainder = n - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)
    for j in range(remainder):
        floors[order[j]] += 1
    return floors


def main() -> None:
    args = parse_args()
    if args.num_samples < 0:
        print("num-samples must be >= 0", file=sys.stderr)
        sys.exit(1)

    counts = load_cs_category_counts(args.counts)
    buckets = build_cs_buckets(args.mapping)

    nonempty = sorted(k for k, items in buckets.items() if items)
    if not nonempty:
        print("No non-empty cs.* buckets after reading mapping.", file=sys.stderr)
        sys.exit(1)

    missing_in_counts = [c for c in nonempty if c not in counts]
    if missing_in_counts:
        print(
            "Warning: these cs.* buckets appear in mapping but not in counts CSV "
            f"(they will be skipped): {missing_in_counts}",
            file=sys.stderr,
        )

    empty_but_counted = sorted(c for c, v in counts.items() if v > 0 and not buckets.get(c))
    if empty_but_counted:
        print(
            "Warning: these cs.* have CSV counts but no leaves in buckets "
            f"(excluded from stratification): {empty_but_counted}",
            file=sys.stderr,
        )

    strata = sorted(c for c in nonempty if c in counts)
    if not strata:
        print("No overlap between nonempty buckets and CSV cs.* counts.", file=sys.stderr)
        sys.exit(1)

    weights = [float(counts[c]) for c in strata]
    alloc = largest_remainder_allocation(args.num_samples, weights)
    total_w = sum(weights)
    rel_weights = [w / total_w for w in weights]

    rng = np.random.default_rng(args.random_seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        sample_idx = 0
        for cat, n_k, rw in zip(strata, alloc, rel_weights):
            items = buckets[cat]
            if n_k == 0:
                continue
            idx = rng.integers(0, len(items), size=n_k, endpoint=False)
            for j in idx:
                acm_id, leaf = items[int(j)]
                rec = {
                    "sample_index": sample_idx,
                    "stratum": cat,
                    "stratum_relative_weight": rw,
                    "stratum_count_csv": counts[cat],
                    "stratum_allocation": int(n_k),
                    "acm_id": acm_id,
                    "leaf_label": leaf,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_idx += 1

    if sample_idx != args.num_samples:
        print(
            f"Warning: wrote {sample_idx} records, expected {args.num_samples}.",
            file=sys.stderr,
        )

    if args.allocation_report is not None:
        report = {
            "num_samples_requested": args.num_samples,
            "random_seed": args.random_seed,
            "strata": [
                {
                    "stratum": c,
                    "csv_count": counts[c],
                    "stratum_relative_weight": w / total_w,
                    "bucket_size": len(buckets[c]),
                    "allocation": int(a),
                }
                for c, w, a in zip(strata, weights, alloc)
            ],
        }
        args.allocation_report.parent.mkdir(parents=True, exist_ok=True)
        args.allocation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""

python sample_stratified_topic_seeds.py \
  -n 20000 \
  --random-seed 42 \
  --output topic_seeds_stratified.jsonl \
  --allocation-report stratified_allocation.json

"""
