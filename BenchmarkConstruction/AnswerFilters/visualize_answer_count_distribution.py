from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from .select_benchmark_queries import difficulty_for_count
except ImportError:
    from select_benchmark_queries import difficulty_for_count  # type: ignore[import-not-found,no-redef]


ANSWER_FILTER_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ANSWER_FILTER_ROOT / "output" / "qwen_final_answer_list_selector_only_status_ok" / "final_answer_list.jsonl"
DEFAULT_OUTPUT_DIR = ANSWER_FILTER_ROOT / "output" / "qwen_final_answer_list_selector_only_status_ok"

BUCKET_LABELS = [
    "Zero",
    "1-4",
    "Easy 5-20",
    "Medium 21-100",
    "Hard 101-150",
    "Over 150",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize final answer count distribution across queries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Final answer list JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for figures.")
    return parser.parse_args()


def configure_plot_style() -> str:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = "Times New Roman" if "Times New Roman" in available_fonts else "Liberation Serif"
    if selected_font != "Times New Roman":
        print("Times New Roman is not installed. Falling back to Liberation Serif.")

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font


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


def extract_answer_count(row: dict[str, Any]) -> int:
    final_answers = row.get("final_answers")
    if isinstance(final_answers, dict) and final_answers.get("final_answer_count") is not None:
        return int(final_answers["final_answer_count"])
    if row.get("final_answer_count") is not None:
        return int(row["final_answer_count"])
    raise ValueError("Row is missing final_answer_count.")


def bucket_for_count(answer_count: int) -> str:
    difficulty = difficulty_for_count(answer_count)
    if difficulty == "easy":
        return "Easy 5-20"
    if difficulty == "medium":
        return "Medium 21-100"
    if difficulty == "hard":
        return "Hard 101-150"
    if answer_count == 0:
        return "Zero"
    if 1 <= answer_count <= 4:
        return "1-4"
    return "Over 150"


def annotate_bars(ax: Any, values: list[int]) -> None:
    max_value = max(values) if values else 0
    for index, value in enumerate(values):
        ax.text(
            index,
            value + max(1, max_value * 0.015),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_distribution(answer_counts: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = list(range(0, max(answer_counts) + 26, 25))
    ax.hist(answer_counts, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.82)
    ax.set_title("Final Answer Count Distribution")
    ax.set_xlabel("Final Answer Count per Query")
    ax.set_ylabel("Query Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "answer_count_distribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_log_distribution(answer_counts: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = list(range(0, max(answer_counts) + 26, 25))
    ax.hist(answer_counts, bins=bins, color="#F58518", edgecolor="black", alpha=0.82)
    ax.set_yscale("log")
    ax.set_title("Final Answer Count Distribution (Log Scale)")
    ax.set_xlabel("Final Answer Count per Query")
    ax.set_ylabel("Query Count (Log Scale)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "answer_count_distribution_log.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_counts(bucket_counts: Counter[str], output_dir: Path) -> None:
    values = [bucket_counts[label] for label in BUCKET_LABELS]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(BUCKET_LABELS, values, color="#54A24B", edgecolor="black", alpha=0.86)
    ax.set_title("Query Count by Final Answer Bucket")
    ax.set_xlabel("Final Answer Bucket")
    ax.set_ylabel("Query Count")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, values)
    fig.tight_layout()
    fig.savefig(output_dir / "answer_count_bucket_distribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    *,
    path: Path,
    answer_counts: list[int],
    bucket_counts: Counter[str],
    selected_font: str,
) -> None:
    nonzero_counts = [count for count in answer_counts if count > 0]
    summary = {
        "input_query_count": len(answer_counts),
        "total_final_answer_references": sum(answer_counts),
        "unique_answer_count_values": len(set(answer_counts)),
        "min_answer_count": min(answer_counts),
        "max_answer_count": max(answer_counts),
        "mean_answer_count": round(mean(answer_counts), 6),
        "median_answer_count": median(answer_counts),
        "zero_answer_query_count": bucket_counts["Zero"],
        "nonzero_answer_query_count": len(nonzero_counts),
        "bucket_counts": {label: bucket_counts[label] for label in BUCKET_LABELS},
        "selected_font": selected_font,
        "figures": {
            "histogram": str(path.parent / "answer_count_distribution.png"),
            "log_histogram": str(path.parent / "answer_count_distribution_log.png"),
            "bucket_distribution": str(path.parent / "answer_count_bucket_distribution.png"),
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_font = configure_plot_style()
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    answer_counts = [extract_answer_count(row) for row in rows]
    bucket_counts = Counter(bucket_for_count(count) for count in answer_counts)

    plot_distribution(answer_counts, output_dir)
    plot_log_distribution(answer_counts, output_dir)
    plot_bucket_counts(bucket_counts, output_dir)
    write_summary(
        path=output_dir / "answer_count_bucket_summary.json",
        answer_counts=answer_counts,
        bucket_counts=bucket_counts,
        selected_font=selected_font,
    )

    print(f"Saved figures to {output_dir}.")
    print(f"Processed query count: {len(answer_counts)}.")


if __name__ == "__main__":
    main()
