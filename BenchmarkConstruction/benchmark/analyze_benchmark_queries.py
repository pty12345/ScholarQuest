from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final


BENCHMARK_FILES: Final[tuple[str, ...]] = (
    "PASA_Benchmark.jsonl",
    "SPAR_Benchmark.jsonl",
)

REPORT_FILE: Final[str] = "benchmark_query_report.md"


@dataclass(frozen=True)
class QueryRecord:
    source_file: str
    line_number: int
    qid: str
    query: str


@dataclass(frozen=True)
class ConstraintRule:
    key: str
    title: str
    description: str
    patterns: tuple[str, ...]


CONSTRAINT_RULES: Final[tuple[ConstraintRule, ...]] = (
    ConstraintRule(
        key="document_type",
        title="Document Type Constraint",
        description="The query restricts the requested output to a specific literature or artifact type.",
        patterns=(
            r"\bsurvey papers?\b",
            r"\bsurveys?\b",
            r"\bjournal papers?\b",
            r"\bdatasets?\b",
            r"\bbenchmarks?\b",
            r"\bresearch papers?\b",
        ),
    ),
    ConstraintRule(
        key="include_or_exclude",
        title="Include / Exclude Constraint",
        description="The query explicitly includes or excludes a subclass of results.",
        patterns=(
            r"\bexclude\b",
            r"\bexcluding\b",
            r"\bwithout\b",
            r"\bdo not use\b",
            r"\bonly\b",
            r"\btrigger-free\b",
        ),
    ),
    ConstraintRule(
        key="scope_exhaustiveness",
        title="Exhaustive Scope Constraint",
        description="The query asks for a complete or near-complete collection rather than a few examples.",
        patterns=(
            r"\ball papers?\b",
            r"\blist all papers?\b",
            r"\bshow me all\b",
            r"\bprovide me with all\b",
            r"\bgive me all\b",
            r"^all papers",
        ),
    ),
    ConstraintRule(
        key="recency_or_quality",
        title="Recency / Venue Quality Constraint",
        description="The query prefers recent, cutting-edge, popular, or top-tier work.",
        patterns=(
            r"\bcutting-edge\b",
            r"\blatest\b",
            r"\bcurrent\b",
            r"\bpopular papers?\b",
            r"\btop-tier\b",
            r"\bstate[- ]of[- ]the[- ]art\b",
        ),
    ),
    ConstraintRule(
        key="comparison_baseline",
        title="Comparison / Baseline Constraint",
        description="The query is framed around outperforming, underperforming, or being between known baselines.",
        patterns=(
            r"\bbetter than\b",
            r"\bharder than\b",
            r"\beasier than\b",
            r"\bperform better than\b",
            r"\bcannot surpass\b",
            r"\bdoes not enhance\b",
            r"\bnegatively impact\b",
        ),
    ),
    ConstraintRule(
        key="threshold_definition",
        title="Threshold / Definition Constraint",
        description="The query gives an explicit threshold, operational definition, or scale condition.",
        patterns=(
            r"\bat least\b",
            r"\bdefined as\b",
            r"\blarge-scale\b",
            r"\bmid-level hardness\b",
            r"\bzero-shot manner\b",
            r"\bdata-scarce\b",
            r"\bnisq\b",
        ),
    ),
    ConstraintRule(
        key="method_or_capability",
        title="Method / Capability Constraint",
        description="The query requires a method, architecture, capability, modality, or training setup to be present.",
        patterns=(
            r"\bthat use\b",
            r"\bthat are\b",
            r"\bthat discuss\b",
            r"\bthat support\b",
            r"\bshould be pre-trained\b",
            r"\busing\b",
            r"\bleveraging\b",
            r"\bbased on\b",
        ),
    ),
    ConstraintRule(
        key="evaluation_setting",
        title="Evaluation / Setting Constraint",
        description="The query anchors the search to a dataset, benchmark, application setting, or experimental environment.",
        patterns=(
            r"\bthrough experiments on\b",
            r"\bon the [a-z0-9][a-z0-9\-]* dataset\b",
            r"\bin the financial sector\b",
            r"\bfor autonomous driving\b",
            r"\bfor machine translation\b",
            r"\bagent tasks?\b",
            r"\bapplication performance\b",
            r"\bdata-scarce scenarios\b",
        ),
    ),
    ConstraintRule(
        key="output_request",
        title="Output Style Constraint",
        description="The query asks for explanation, analysis, or a specific answer presentation style in addition to paper retrieval.",
        patterns=(
            r"\bplease explain\b",
            r"\bcomprehensive analysis\b",
            r"\bin detail\b",
            r"\bshare some insights\b",
            r"\bwith supporting research papers\b",
            r"\brelated papers\b",
        ),
    ),
)


def load_queries(benchmark_dir: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for filename in BENCHMARK_FILES:
        path = benchmark_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                query = str(
                    payload.get("question")
                    or payload.get("query")
                    or payload.get("prompt")
                    or payload.get("input")
                    or ""
                ).strip()
                if not query:
                    continue
                records.append(
                    QueryRecord(
                        source_file=filename,
                        line_number=line_number,
                        qid=str(payload.get("qid", f"{filename}:{line_number}")),
                        query=normalize_query(query),
                    )
                )
    return records


def normalize_query(query: str) -> str:
    collapsed = " ".join(query.replace("\n", " ").split())
    return collapsed.strip()


def classify_constraints(query: str) -> list[str]:
    lowered = query.lower()
    matches: list[str] = []
    for rule in CONSTRAINT_RULES:
        if any(re.search(pattern, lowered) for pattern in rule.patterns):
            matches.append(rule.key)
    return matches


def detect_surface_style(query: str) -> list[str]:
    styles: list[str] = []
    stripped = query.strip()
    first_token = stripped.split()[0].lower() if stripped.split() else ""

    if first_token in {"show", "give", "provide", "find", "list", "search"}:
        styles.append("imperative_retrieval")
    if first_token in {"how", "what", "why", "can", "is", "do"}:
        styles.append("question_form")
    if stripped.count("?") >= 1:
        styles.append("contains_question_mark")
    if stripped.count(".") >= 2 or stripped.count("?") >= 2:
        styles.append("multi_sentence_or_multi_clause")
    if len(stripped.split()) >= 25:
        styles.append("long_form_query")
    return styles


def summarize_by_source(records: list[QueryRecord]) -> dict[str, dict[str, int]]:
    per_source: dict[str, dict[str, int]] = {}
    for filename in BENCHMARK_FILES:
        source_records = [record for record in records if record.source_file == filename]
        style_counter: Counter[str] = Counter()
        for record in source_records:
            for style in detect_surface_style(record.query):
                style_counter[style] += 1
        per_source[filename] = {
            "count": len(source_records),
            "question_marks": sum(1 for record in source_records if "?" in record.query),
            "multi_sentence_or_multi_clause": style_counter["multi_sentence_or_multi_clause"],
            "imperative_retrieval": style_counter["imperative_retrieval"],
            "question_form": style_counter["question_form"],
            "long_form_query": style_counter["long_form_query"],
        }
    return per_source


def build_report(records: list[QueryRecord]) -> str:
    record_constraints = {record.qid: classify_constraints(record.query) for record in records}

    overall_counter: Counter[str] = Counter()
    source_counter: dict[str, Counter[str]] = {filename: Counter() for filename in BENCHMARK_FILES}
    examples: dict[str, list[QueryRecord]] = defaultdict(list)

    for record in records:
        labels = record_constraints[record.qid]
        for label in labels:
            overall_counter[label] += 1
            source_counter[record.source_file][label] += 1
            if len(examples[label]) < 4:
                examples[label].append(record)

    uncategorized = [record for record in records if not record_constraints[record.qid]]
    source_summary = summarize_by_source(records)

    lines: list[str] = []
    lines.append("# Benchmark Query Constraint Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Source files: `{BENCHMARK_FILES[0]}`, `{BENCHMARK_FILES[1]}`")
    lines.append(f"- Total extracted queries: `{len(records)}`")
    lines.append("- Focus: classify surface-level query constraints rather than research topics")
    lines.append("- Method: rule-based multi-label tagging with manual taxonomy design")
    lines.append("")
    lines.append("## High-Level Findings")
    lines.append("")
    lines.append("- The benchmark queries usually mix topic intent with retrieval constraints, evidence requirements, and answer-format instructions.")
    lines.append("- `PASA_Benchmark.jsonl` is closer to imperative search requests, while `SPAR_Benchmark.jsonl` is closer to open-form research questions.")
    lines.append("- Constraint composition is uneven: a small number of constraints appear often, while many queries remain mostly topic-driven and lightly constrained.")
    lines.append("- For future benchmark design, it is useful to separate topic diversity from constraint diversity and control both explicitly.")
    lines.append("")
    lines.append("## Source-Level Style Comparison")
    lines.append("")
    for filename in BENCHMARK_FILES:
        summary = source_summary[filename]
        lines.append(f"### `{filename}`")
        lines.append("")
        lines.append(f"- Query count: `{summary['count']}`")
        lines.append(f"- Imperative retrieval style: `{summary['imperative_retrieval']}`")
        lines.append(f"- Question form: `{summary['question_form']}`")
        lines.append(f"- Contains question mark: `{summary['question_marks']}`")
        lines.append(f"- Multi-sentence or multi-clause queries: `{summary['multi_sentence_or_multi_clause']}`")
        lines.append(f"- Long-form queries: `{summary['long_form_query']}`")
        lines.append("")

    lines.append("## Constraint Taxonomy")
    lines.append("")
    for rule in CONSTRAINT_RULES:
        total = overall_counter[rule.key]
        pasa_count = source_counter["PASA_Benchmark.jsonl"][rule.key]
        spar_count = source_counter["SPAR_Benchmark.jsonl"][rule.key]
        lines.append(f"### {rule.title}")
        lines.append("")
        lines.append(f"- Definition: {rule.description}")
        lines.append(f"- Overall count: `{total}`")
        lines.append(f"- `PASA_Benchmark.jsonl`: `{pasa_count}`")
        lines.append(f"- `SPAR_Benchmark.jsonl`: `{spar_count}`")
        lines.append("- Representative queries:")
        for record in examples[rule.key]:
            lines.append(
                f"  - `{record.source_file}` / `{record.qid}`: {record.query}"
            )
        if not examples[rule.key]:
            lines.append("  - None")
        lines.append("")

    lines.append("## Distribution Notes")
    lines.append("")
    most_common = overall_counter.most_common()
    for label, count in most_common:
        title = next(rule.title for rule in CONSTRAINT_RULES if rule.key == label)
        lines.append(f"- `{title}` appears in `{count}` queries.")
    if not most_common:
        lines.append("- No constraints were detected.")
    lines.append(f"- Queries without any current rule match: `{len(uncategorized)}`")
    lines.append("")

    lines.append("## Design Implications For New Benchmark Queries")
    lines.append("")
    lines.append("- Separate the topic slot from the constraint slot. A future query template should let you compose them independently.")
    lines.append("- Add explicit contrastive buckets such as `include only`, `exclude`, `between baselines`, and `threshold-defined scope` because they are frequent but not evenly covered.")
    lines.append("- Keep both imperative search requests and research-question style prompts. They stress retrieval systems differently.")
    lines.append("- Avoid mixing too many constraints in a single query unless that is an intentional hard setting. Some existing queries combine topic, modality, scale, exclusion, and answer-style requirements at once.")
    lines.append("- Track whether a constraint is retrieval-facing or answer-format-facing. These are different difficulty sources.")
    lines.append("- Consider building a balanced benchmark matrix: `topic family x constraint family x surface form`.")
    lines.append("")

    lines.append("## Suggested Constraint Families For Future Query Authoring")
    lines.append("")
    lines.append("- `Document type`: survey, dataset, benchmark, journal paper, workshop paper")
    lines.append("- `Include / exclude`: only, except, without, exclude")
    lines.append("- `Scope`: all papers, representative papers, a few examples, exhaustive list")
    lines.append("- `Time / quality`: latest, classic, top-tier, popular, influential")
    lines.append("- `Comparison`: better than, worse than, harder than, between A and B")
    lines.append("- `Definition / threshold`: at least N, long-form defined as X, large-scale defined as Y")
    lines.append("- `Method / capability`: must use RLHF, must support audio, must be autoregressive")
    lines.append("- `Evaluation / setting`: on HotPotQA, in finance, for robotics, under data scarcity")
    lines.append("- `Output style`: explain why, provide analysis, summarize evidence")
    lines.append("")

    lines.append("## Full Extracted Query Inventory")
    lines.append("")
    current_source = ""
    for record in records:
        if record.source_file != current_source:
            current_source = record.source_file
            lines.append(f"### `{current_source}`")
            lines.append("")
        labels = record_constraints[record.qid]
        label_text = ", ".join(labels) if labels else "none"
        lines.append(f"- `{record.qid}`")
        lines.append(f"  - Query: {record.query}")
        lines.append(f"  - Constraint labels: `{label_text}`")
    lines.append("")

    lines.append("## Queries Needing Taxonomy Expansion")
    lines.append("")
    lines.append("- These queries are valid, but the current rule set does not yet capture them well. They are good candidates for future taxonomy refinement.")
    lines.append("")
    for record in uncategorized:
        lines.append(f"- `{record.source_file}` / `{record.qid}`: {record.query}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    benchmark_dir = Path(__file__).resolve().parent
    records = load_queries(benchmark_dir)
    report = build_report(records)
    report_path = benchmark_dir / REPORT_FILE
    report_path.write_text(report, encoding="utf-8")
    print(f"Loaded {len(records)} queries from {len(BENCHMARK_FILES)} files.")
    print(f"Report written to {report_path.name}.")


if __name__ == "__main__":
    main()
