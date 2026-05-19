"""Report ACM CCS2012 leaf seeds with no arXiv cs.* mapping (empty matched_categories)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "List JSONL records where matched_categories is empty (no cs.* bucket assigned)."
        )
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=here / "acm_ccs2012_leaf_to_arxiv_cs.jsonl",
        help="ACM leaf -> arXiv cs JSONL.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=here / "empty_matched_categories_seeds.json",
        help="Write JSON report (count + seed list).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Only print summary; do not write --output.",
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
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def main() -> int:
    args = parse_args()
    mapping_path: Path = args.mapping
    if not mapping_path.is_file():
        print(f"error: mapping file not found: {mapping_path}", file=sys.stderr)
        return 1

    empty_seeds: list[dict[str, Any]] = []
    total = 0
    for row in iter_jsonl(mapping_path):
        total += 1
        matched = row.get("matched_categories")
        if not isinstance(matched, list):
            continue
        if len(matched) != 0:
            continue
        empty_seeds.append(
            {
                "acm_id": row.get("acm_id"),
                "leaf_label": row.get("leaf_label"),
                "depth": row.get("depth"),
                "path_text": row.get("path_text"),
                "primary_category": row.get("primary_category"),
                "discarded": row.get("discarded"),
                "reason": row.get("reason"),
            }
        )

    print(f"total_rows={total} empty_matched_categories={len(empty_seeds)}")
    if not args.no_write:
        out: Path = args.output
        payload = {
            "source": str(mapping_path.resolve()),
            "definition": "matched_categories is empty (no cs.* category assigned)",
            "total_rows": total,
            "empty_count": len(empty_seeds),
            "seeds": empty_seeds,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
