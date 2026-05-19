from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSWER_FILTER_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ANSWER_FILTER_ROOT
    / "output"
    / "qwen_final_answer_list_first347_first500"
    / "final_answer_list.jsonl"
)
DEFAULT_OUTPUT_DIR = ANSWER_FILTER_ROOT / "output" / "qwen_final_answer_list_first347_first500"
DEFAULT_SELECTION_OUTPUT = DEFAULT_OUTPUT_DIR / "benchmark_query_selection.jsonl"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "benchmark_query_selection_summary.json"

DIFFICULTY_QUOTAS = {
    "easy": 30,
    "medium": 90,
    "hard": 30,
}

TYPE_CAP_RATIOS = {
    "comparative_claim": 0.10,
    "negative_constraint": 0.28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced PaperSearch benchmark queries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Final answer list JSONL.")
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT,
        help="Selected benchmark query JSONL.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Selection summary JSON.",
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
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_query_type(final_query: str) -> str:
    query = final_query.lower()
    if any(marker in query for marker in ("outperform", "more secure than", "claim that", "report that")):
        return "comparative_claim"
    if any(
        marker in query
        for marker in (
            "dataset",
            "benchmark",
            "coco",
            "cifar",
            "mot17",
            "davis",
            "cnn/daily mail",
            "starcraft ii",
            "20 newsgroups",
            "ms marco",
            "kitti",
            "kodak",
            "conll-2003",
        )
    ):
        return "dataset_or_benchmark"
    if any(marker in query for marker in (" exclude ", "excluding ", "do not focus", "do not involve")):
        return "negative_constraint"
    if any(marker in query for marker in (" using ", "use of ", "trained with ", " with ", "focus on")):
        return "method_or_technique"
    if any(
        marker in query
        for marker in (
            "domain",
            "context",
            "applications",
            "robotics",
            "medical",
            "automotive",
            "healthcare",
        )
    ):
        return "application_or_domain"
    return "general_topical"


def difficulty_for_count(answer_count: int) -> str | None:
    if 5 <= answer_count <= 20:
        return "easy"
    if 21 <= answer_count <= 100:
        return "medium"
    if 101 <= answer_count <= 150:
        return "hard"
    return None


def template_family(final_query: str) -> str:
    query = final_query.lower()
    if query.startswith("i am looking for papers on") and any(
        marker in query for marker in (" exclude ", "excluding ", "do not focus", "do not involve")
    ):
        return "looking_for_negative"
    if query.startswith("which papers evaluate"):
        return "evaluate"
    if query.startswith("which papers study"):
        return "study"
    if query.startswith("which papers explore"):
        return "explore"
    if query.startswith("which papers apply"):
        return "apply"
    if query.startswith("which papers report") or query.startswith("which papers claim"):
        return "claim_report"
    return "other"


def extract_candidate(row: dict[str, Any]) -> dict[str, Any]:
    query = row["query"]
    final_answers = row["final_answers"]
    answer_counts = row["answer_counts"]
    answer_count = int(final_answers["final_answer_count"])
    final_query = str(query["final_query"])
    query_type = classify_query_type(final_query)
    difficulty = difficulty_for_count(answer_count)
    return {
        "query_id": query["query_id"],
        "final_query": final_query,
        "domain": query.get("domain"),
        "category": query.get("category"),
        "constraint_kind": query.get("constraint_kind"),
        "constraint_value": query.get("constraint_value"),
        "topic_seed": query.get("topic_seed"),
        "risk_flags": query.get("risk_flags", []),
        "final_answer_count": answer_count,
        "final_answer_arxiv_ids": final_answers.get("final_answer_arxiv_ids", []),
        "qwen_score_2_count": answer_counts.get("qwen_score_2_count"),
        "qwen_source": "first347_first500",
        "difficulty": difficulty,
        "query_type": query_type,
        "template_family": template_family(final_query),
        "selection_notes": [],
    }


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    type_priority = {
        "dataset_or_benchmark": 0,
        "application_or_domain": 1,
        "method_or_technique": 2,
        "general_topical": 3,
        "negative_constraint": 4,
        "comparative_claim": 5,
    }
    return sorted(
        candidates,
        key=lambda item: (
            bool(item["risk_flags"]),
            type_priority.get(item["query_type"], 99),
            item["template_family"] == "looking_for_negative",
            -item["final_answer_count"],
            item["query_id"],
        ),
    )


def selection_penalty(candidate: dict[str, Any], selected: list[dict[str, Any]], target_total: int) -> tuple[int, int, int, int, int, str]:
    type_counts = Counter(item["query_type"] for item in selected)
    domain_counts = Counter(item["domain"] for item in selected)
    template_counts = Counter(item["template_family"] for item in selected)
    topic_counts = Counter(item["topic_seed"] for item in selected)

    query_type = candidate["query_type"]
    cap = int(TYPE_CAP_RATIOS.get(query_type, 1.0) * target_total)
    type_cap_penalty = 5 if type_counts[query_type] >= cap else 0
    risk_penalty = 3 if candidate["risk_flags"] else 0
    return (
        type_cap_penalty,
        domain_counts[candidate["domain"]],
        template_counts[candidate["template_family"]],
        topic_counts[candidate["topic_seed"]],
        risk_penalty,
        candidate["query_id"],
    )


def select_balanced(candidates: list[dict[str, Any]], quota: int, target_total: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = sort_candidates(candidates)
    used_ids: set[str] = set()

    while len(selected) < quota and remaining:
        remaining = [item for item in remaining if item["query_id"] not in used_ids]
        if not remaining:
            break
        best = min(remaining, key=lambda item: selection_penalty(item, selected, target_total))
        selected.append(best)
        used_ids.add(best["query_id"])

    return selected


def build_summary(all_candidates: list[dict[str, Any]], selected: list[dict[str, Any]], backfill_count: int) -> dict[str, Any]:
    selected_counts = [item["final_answer_count"] for item in selected]
    excluded_zero = sum(item["final_answer_count"] == 0 for item in all_candidates)
    excluded_too_broad = sum(item["final_answer_count"] > 150 for item in all_candidates)
    eligible = [item for item in all_candidates if item["difficulty"] is not None]
    return {
        "input_query_count": len(all_candidates),
        "selected_query_count": len(selected),
        "target_query_count": sum(DIFFICULTY_QUOTAS.values()),
        "backfill_from_high_medium_count": backfill_count,
        "excluded_zero_answer_count": excluded_zero,
        "excluded_answer_count_over_150": excluded_too_broad,
        "eligible_query_count": len(eligible),
        "difficulty_quotas": DIFFICULTY_QUOTAS,
        "selected_difficulty_counts": dict(Counter(item["difficulty"] for item in selected)),
        "selected_query_type_counts": dict(Counter(item["query_type"] for item in selected)),
        "selected_domain_counts": dict(Counter(item["domain"] for item in selected)),
        "selected_category_counts": dict(Counter(item["category"] for item in selected)),
        "selected_template_family_counts": dict(Counter(item["template_family"] for item in selected)),
        "selected_answer_count_min": min(selected_counts) if selected_counts else None,
        "selected_answer_count_max": max(selected_counts) if selected_counts else None,
        "selected_answer_count_average": round(sum(selected_counts) / len(selected_counts), 2) if selected_counts else None,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    candidates = [extract_candidate(row) for row in rows]

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["difficulty"] is not None:
            by_difficulty[candidate["difficulty"]].append(candidate)

    selected_by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for difficulty, quota in DIFFICULTY_QUOTAS.items():
        selected_by_difficulty[difficulty] = select_balanced(
            by_difficulty[difficulty],
            quota,
            sum(DIFFICULTY_QUOTAS.values()),
        )

    hard_shortfall = DIFFICULTY_QUOTAS["hard"] - len(selected_by_difficulty["hard"])
    backfill: list[dict[str, Any]] = []
    if hard_shortfall > 0:
        selected_ids = {item["query_id"] for group in selected_by_difficulty.values() for item in group}
        high_medium_pool = [
            item
            for item in by_difficulty["medium"]
            if item["query_id"] not in selected_ids and 81 <= item["final_answer_count"] <= 100
        ]
        backfill = select_balanced(high_medium_pool, hard_shortfall, sum(DIFFICULTY_QUOTAS.values()))
        for item in backfill:
            item["selection_notes"].append("backfilled_from_high_medium_due_to_hard_shortfall")

    selected = (
        selected_by_difficulty["easy"]
        + selected_by_difficulty["medium"]
        + selected_by_difficulty["hard"]
        + backfill
    )
    selected = sorted(selected, key=lambda item: (item["difficulty"] or "medium", item["query_id"]))

    write_jsonl(args.selection_output, selected)
    write_json(args.summary_output, build_summary(candidates, selected, len(backfill)))
    print(f"Selected {len(selected)} benchmark queries.")
    print(f"Selection output: {args.selection_output}")
    print(f"Summary output: {args.summary_output}")


if __name__ == "__main__":
    main()
