#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count arXiv paper categories from the metadata snapshot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("arxiv-metadata-oai-snapshot.json"),
        help="Path to the JSON Lines metadata file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "first"],
        default="all",
        help="Use all categories for each paper, or only the first category.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top categories to keep in the summary file.",
    )
    return parser.parse_args()


def iter_categories(category_text: str, mode: str) -> list[str]:
    categories = category_text.split()
    if mode == "first":
        return categories[:1]
    return categories


def count_categories(input_path: Path, mode: str) -> tuple[Counter, int, int, int]:
    counter = Counter()
    total_papers = 0
    missing_categories = 0
    multi_label_papers = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            total_papers += 1
            record = json.loads(line)
            category_text = record.get("categories")
            if not category_text:
                missing_categories += 1
                continue

            raw_categories = category_text.split()
            if len(raw_categories) > 1:
                multi_label_papers += 1

            counter.update(iter_categories(category_text, mode))

    return counter, total_papers, missing_categories, multi_label_papers


def write_outputs(
    output_dir: Path,
    input_path: Path,
    mode: str,
    top_k: int,
    counter: Counter,
    total_papers: int,
    missing_categories: int,
    multi_label_papers: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "all" if mode == "all" else "first"
    json_path = output_dir / f"category_counts_{mode_suffix}.json"
    csv_path = output_dir / f"category_counts_{mode_suffix}.csv"
    summary_path = output_dir / f"category_counts_{mode_suffix}_summary.json"

    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    summary = {
        "statistic_mode": "all_categories" if mode == "all" else "first_category_only",
        "input_file": input_path.name,
        "total_papers": total_papers,
        "missing_categories": missing_categories,
        "multi_label_papers": multi_label_papers,
        "multi_label_ratio": (
            multi_label_papers / (total_papers - missing_categories)
            if total_papers > missing_categories
            else 0
        ),
        "unique_categories": len(counter),
        "top_categories": [
            {"category": category, "count": count}
            for category, count in sorted_items[:top_k]
        ],
    }

    full_result = {
        **summary,
        "category_counts": [
            {"category": category, "count": count}
            for category, count in sorted_items
        ],
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(full_result, file, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "count"])
        writer.writerows(sorted_items)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return json_path, csv_path, summary_path


def main() -> None:
    args = parse_args()

    counter, total_papers, missing_categories, multi_label_papers = count_categories(
        args.input, args.mode
    )
    json_path, csv_path, summary_path = write_outputs(
        output_dir=args.output_dir,
        input_path=args.input,
        mode=args.mode,
        top_k=args.top_k,
        counter=counter,
        total_papers=total_papers,
        missing_categories=missing_categories,
        multi_label_papers=multi_label_papers,
    )

    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    print(f"statistic_mode={args.mode}")
    print(f"total_papers={total_papers}")
    print(f"missing_categories={missing_categories}")
    print(f"multi_label_papers={multi_label_papers}")
    print(f"unique_categories={len(counter)}")
    print("top_10_categories=")
    for category, count in sorted_items[:10]:
        print(f"{category},{count}")
    print(f"json_out={json_path}")
    print(f"csv_out={csv_path}")
    print(f"summary_out={summary_path}")


if __name__ == "__main__":
    main()
