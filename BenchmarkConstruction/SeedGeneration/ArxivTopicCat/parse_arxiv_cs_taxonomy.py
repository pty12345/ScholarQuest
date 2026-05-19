from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("Category Taxonomy.html")
DEFAULT_OUTPUT = Path(__file__).with_name("cs_taxonomy.jsonl")

CS_PANEL_ID = 'id="accordion-panel-grp_cs"'
NEXT_GROUP_MARKER = '<h2 class="accordion-head" id="accordion-head-grp_'
ENTRY_PATTERN = re.compile(
    r"<h4>\s*(cs\.[A-Z]{2})\s*<span>\((.*?)\)</span>\s*</h4>\s*</div>\s*"
    r'<div class="column"><p>(.*?)</p></div>',
    re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class CategoryRecord:
    code: str
    name: str
    description: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "code": self.code,
                "name": self.name,
                "description": self.description,
            },
            ensure_ascii=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract arXiv Computer Science taxonomy records from HTML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Category Taxonomy HTML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output JSONL file.",
    )
    return parser.parse_args()


def normalize_html_text(value: str) -> str:
    text = unescape(value)
    text = TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def extract_cs_section(html_text: str) -> str:
    start_index = html_text.find(CS_PANEL_ID)
    if start_index == -1:
        raise ValueError("Could not find the Computer Science taxonomy panel.")

    next_group_index = html_text.find(NEXT_GROUP_MARKER, start_index + len(CS_PANEL_ID))
    if next_group_index == -1:
        raise ValueError("Could not find the end of the Computer Science taxonomy panel.")

    return html_text[start_index:next_group_index]


def parse_cs_categories(html_text: str) -> list[CategoryRecord]:
    cs_section = extract_cs_section(html_text)
    records = [
        CategoryRecord(
            code=code,
            name=normalize_html_text(name),
            description=normalize_html_text(description),
        )
        for code, name, description in ENTRY_PATTERN.findall(cs_section)
    ]
    if not records:
        raise ValueError("No Computer Science taxonomy records were extracted.")
    return records


def write_jsonl(output_path: Path, records: list[CategoryRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")


def main() -> None:
    args = parse_args()
    html_text = args.input.read_text(encoding="utf-8")
    records = parse_cs_categories(html_text)
    write_jsonl(args.output, records)

    print(f"Extracted {len(records)} Computer Science categories.")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
