from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Count duplicate topic seeds in a JSONL file and report frequencies / "
            "empirical duplicate proportions."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=here / "topic_seeds_stratified.jsonl",
        help="Topic seed JSONL (expects acm_id and leaf_label per line).",
    )
    p.add_argument(
        "--key",
        choices=("acm_leaf", "stratum_acm_leaf"),
        default="acm_leaf",
        help=(
            "Seed identity: acm_id+leaf_label (default), or stratum+acm_id+leaf_label "
            "(treat same ACM leaf in different strata as distinct)."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many most frequent seeds to print.",
    )
    p.add_argument(
        "--csv-summary",
        type=Path,
        default=None,
        help="Optional path to write a short one-row CSV summary.",
    )
    return p.parse_args()


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


def seed_key(row: dict[str, Any], mode: str) -> tuple[str, ...]:
    acm = str(row.get("acm_id", ""))
    leaf = str(row.get("leaf_label", ""))
    if mode == "acm_leaf":
        return (acm, leaf)
    st = str(row.get("stratum", ""))
    return (st, acm, leaf)


def multiplicity_histogram(counts: Counter[tuple[str, ...]]) -> dict[int, int]:
    """How many distinct seeds have multiplicity k."""
    hist: dict[int, int] = {}
    for _key, c in counts.items():
        hist[c] = hist.get(c, 0) + 1
    return dict(sorted(hist.items()))


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    counts: Counter[tuple[str, ...]] = Counter()
    n = 0
    for row in iter_jsonl(args.input):
        counts[seed_key(row, args.key)] += 1
        n += 1

    if n == 0:
        print("No rows read.", file=sys.stderr)
        sys.exit(1)

    u = len(counts)
    redundant = n - u
    frac_redundant = redundant / n
    seeds_with_duplicates = sum(1 for c in counts.values() if c > 1)
    frac_keys_dup = seeds_with_duplicates / u

    print(f"Input: {args.input}")
    print(f"Identity key: {args.key}")
    print(f"Total samples N: {n}")
    print(f"Distinct seeds U: {u}")
    print(f"Redundant draws (N - U): {redundant}")
    print(f"P_redundant_draw = (N-U)/N: {frac_redundant:.6f}")
    print(f"Distinct seeds with count>1: {seeds_with_duplicates}")
    print(f"P_dup_key among distinct = dup_keys/U: {frac_keys_dup:.6f}")
    print("Multiplicity histogram (k -> number of distinct seeds with count=k):")
    for k, m in multiplicity_histogram(counts).items():
        print(f"  k={k}: {m}")

    top = counts.most_common(args.top_k)
    print(f"Top {len(top)} seeds by frequency:")
    for rank, (key, c) in enumerate(top, start=1):
        if args.key == "acm_leaf":
            acm, leaf = key
            print(f"  {rank}. count={c} acm_id={acm} leaf_label={leaf!r}")
        else:
            st, acm, leaf = key
            print(
                f"  {rank}. count={c} stratum={st} acm_id={acm} leaf_label={leaf!r}"
            )

    if args.csv_summary is not None:
        args.csv_summary.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "input_path",
            "key_mode",
            "N",
            "U",
            "redundant_draws",
            "P_redundant_draw",
            "dup_distinct_keys",
            "P_dup_key_among_distinct",
        ]
        row = {
            "input_path": str(args.input),
            "key_mode": args.key,
            "N": n,
            "U": u,
            "redundant_draws": redundant,
            "P_redundant_draw": f"{frac_redundant:.10f}",
            "dup_distinct_keys": seeds_with_duplicates,
            "P_dup_key_among_distinct": f"{frac_keys_dup:.10f}",
        }
        with args.csv_summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()

"""
python analyze_topic_seed_duplicates.py \
  --input topic_seeds_stratified.jsonl \
  --top-k 30 \
  --csv-summary duplicate_summary.csv
"""