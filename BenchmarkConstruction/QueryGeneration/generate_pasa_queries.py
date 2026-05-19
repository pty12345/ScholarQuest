from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QUERY_GENERATION_DIR = Path(__file__).resolve().parent
if str(QUERY_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUERY_GENERATION_DIR))

from paperbench.io_utils import append_jsonl, ensure_parent, read_jsonl
from paperbench.llm.client import OpenAICompatibleClient
from pasa_types import (
    PasaCategory,
    PasaExpansionPlan,
    PasaGenerationConfig,
    PasaGenerationTrace,
    PasaPromptConfig,
    PasaQueryDraftResponse,
    PasaQueryRecord,
    PasaRewriteResponse,
    PasaTopicSeed,
    PasaValidationResponse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PASA-style queries from topic seeds.")
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
        "--seed-file",
        type=Path,
        default=None,
        help="Optional override for the input seed JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional override for the output query JSONL file.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="Optional override for the trace JSONL file.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional override for the summary report JSON file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based start index after seed deduplication.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of deduplicated seeds to process.",
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


def _load_config(config_path: Path) -> PasaGenerationConfig:
    return PasaGenerationConfig(**_load_yaml_mapping(config_path))


def _load_prompts(prompt_path: Path) -> PasaPromptConfig:
    return PasaPromptConfig(**_load_yaml_mapping(prompt_path))


def _load_llm_client(
    *,
    root_dir: Path,
    config: PasaGenerationConfig,
    cache_path: Path,
) -> tuple[OpenAICompatibleClient, str]:
    load_dotenv(root_dir / ".env")
    api_key = os.getenv(config.llm.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key from environment variable {config.llm.api_key_env}")

    llm_mode = config.llm.mode.strip().lower()
    if llm_mode not in {"realtime", "batch_chat"}:
        raise ValueError("llm.mode must be either 'realtime' or 'batch_chat'.")

    if llm_mode == "batch_chat":
        base_url = os.getenv(config.llm.batch_base_url_env, config.llm.batch_base_url)
        use_responses_api = False
    else:
        base_url = os.getenv(config.llm.base_url_env, "")
        use_responses_api = True

    model = os.getenv(config.llm.model_env, config.llm.model)
    ensure_parent(cache_path)
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        cache_path=cache_path,
        temperature=config.llm.temperature,
        use_responses_api=use_responses_api,
    )
    return client, model


def _iter_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(cast(dict[str, object], payload))
    return rows


def _parse_model_from_json_line(line: str, model_type: type[BaseModel]) -> BaseModel:
    validator = getattr(model_type, "model_validate_json", None)
    if callable(validator):
        return cast(BaseModel, validator(line))
    parser = getattr(model_type, "parse_raw", None)
    if callable(parser):
        return cast(BaseModel, parser(line))
    raise TypeError(f"Model type {model_type.__name__} does not support JSON parsing.")


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _opening_signature(query: str, max_words: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z]+", query.lower())
    if not tokens:
        return ""
    return " ".join(tokens[:max_words])


def _collect_recent_openings(records: list[PasaQueryRecord], limit: int = 8) -> list[str]:
    openings: list[str] = []
    for record in reversed(records):
        opening = _opening_signature(record.final_query)
        if not opening:
            continue
        openings.append(opening)
        if len(openings) >= limit:
            break
    openings.reverse()
    return openings


def _opening_counter(openings: list[str]) -> Counter[str]:
    return Counter(opening for opening in openings if opening)


def _looks_templatic(query: str, recent_openings: list[str]) -> bool:
    opening = _opening_signature(query)
    if not opening:
        return False
    repeated_openings = _opening_counter(recent_openings)
    if opening == "find papers":
        return repeated_openings[opening] >= 1
    return repeated_openings[opening] >= 2


def _surface_quality_issues(query: str, recent_openings: list[str]) -> list[str]:
    normalized = " ".join(query.strip().lower().split())
    issues: list[str] = []
    if _looks_templatic(query, recent_openings):
        issues.append("templatic_opening")
    temporal_patterns = (
        r"\brecent\b",
        r"\blatest\b",
        r"\bsince\s+20\d{2}\b",
        r"\bafter\s+20\d{2}\b",
        r"\bbefore\s+20\d{2}\b",
        r"\bpost-?20\d{2}\b",
    )
    if any(re.search(pattern, normalized) for pattern in temporal_patterns):
        issues.append("temporal_wording")
    fragment_prefixes = (
        "papers on ",
        "research on ",
        "studies on ",
        "work on ",
    )
    if any(normalized.startswith(prefix) for prefix in fragment_prefixes):
        issues.append("fragment_opening")
    return issues


def _build_surface_rewrite_instruction(issues: list[str], recent_openings: list[str]) -> str:
    instructions: list[str] = []
    if "templatic_opening" in issues:
        opening_examples = ", ".join(recent_openings[-4:]) if recent_openings else ""
        instructions.append(
            "Use a different natural opening and avoid these recent openings: "
            f"{opening_examples}"
        )
    if "temporal_wording" in issues:
        instructions.append("Remove temporal wording such as recent or latest.")
    if "fragment_opening" in issues:
        instructions.append("Rewrite it as a complete natural request or question, not a fragment.")
    return " ".join(instructions).strip()


def _override_validation_for_surface_quality(
    validation: PasaValidationResponse,
    query: str,
    recent_openings: list[str],
) -> PasaValidationResponse:
    if not validation.is_valid:
        return validation
    issues = _surface_quality_issues(query, recent_openings)
    if not issues:
        return validation
    merged_issues = list(validation.issues)
    for issue in issues:
        if issue not in merged_issues:
            merged_issues.append(issue)
    rewrite_instruction = _build_surface_rewrite_instruction(issues, recent_openings)
    return PasaValidationResponse(
        is_valid=False,
        category_match=validation.category_match,
        retrieval_oriented=validation.retrieval_oriented,
        executable_constraint=validation.executable_constraint,
        natural_language_ok=False,
        issues=merged_issues,
        rewrite_needed=True,
        rewrite_instruction=rewrite_instruction,
        rationale="The query is semantically valid but has surface-level wording issues.",
    )


def _read_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return _normalize_text(value)
    return ""


def _read_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _read_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_topic_seeds(seed_path: Path) -> list[PasaTopicSeed]:
    seeds_by_key: dict[tuple[str, str, str], PasaTopicSeed] = {}
    ordered_keys: list[tuple[str, str, str]] = []

    for row in _iter_jsonl_objects(seed_path):
        stratum = _read_str(row, "stratum")
        acm_id = _read_str(row, "acm_id")
        leaf_label = _read_str(row, "leaf_label")
        if not stratum or not acm_id or not leaf_label:
            continue

        dedup_key = (stratum, acm_id, leaf_label.lower())
        existing = seeds_by_key.get(dedup_key)
        if existing is None:
            ordered_keys.append(dedup_key)
            seeds_by_key[dedup_key] = PasaTopicSeed(
                seed_id="",
                topic=leaf_label,
                domain=stratum,
                acm_id=acm_id,
                sample_index=_read_int(row, "sample_index"),
                stratum_relative_weight=_read_float(row, "stratum_relative_weight"),
                stratum_count_csv=_read_int(row, "stratum_count_csv"),
                stratum_allocation=_read_int(row, "stratum_allocation"),
                source_row_count=1,
                source_stratum=stratum,
                source_leaf_label=leaf_label,
            )
            continue

        existing.source_row_count += 1
        if existing.sample_index is None:
            existing.sample_index = _read_int(row, "sample_index")

    seeds: list[PasaTopicSeed] = []
    for index, dedup_key in enumerate(ordered_keys, start=1):
        seed = seeds_by_key[dedup_key]
        seeds.append(seed.copy(update={"seed_id": f"PTS_{index:06d}"}))
    return seeds


def _slice_seeds(seeds: list[PasaTopicSeed], start: int, limit: int | None) -> list[PasaTopicSeed]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    if limit is None:
        return seeds[start:]
    return seeds[start : start + limit]


def _make_query_id(index: int) -> str:
    return f"PQ_{index:06d}"


def _make_trace_id(index: int) -> str:
    return f"PT_{index:06d}"


def _extract_numeric_suffix(identifier: str) -> int:
    try:
        return int(identifier.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def _load_existing_records(output_path: Path) -> list[PasaQueryRecord]:
    if not output_path.exists():
        return []
    records: list[PasaQueryRecord] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            parsed = _parse_model_from_json_line(raw_line, PasaQueryRecord)
            records.append(cast(PasaQueryRecord, parsed))
    return records


def _load_processed_pairs(records: list[PasaQueryRecord]) -> set[tuple[str, PasaCategory]]:
    return {(record.topic_seed.seed_id, record.category) for record in records}


def _next_record_index(records: list[PasaQueryRecord]) -> int:
    if not records:
        return 1
    return max(_extract_numeric_suffix(record.query_id) for record in records) + 1


def _append_model_jsonl(path: Path, model: BaseModel) -> None:
    append_jsonl(path, cast(dict[str, object], model.dict()))


def _serialize_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_plan_payload(seed: PasaTopicSeed, category: PasaCategory) -> dict[str, object]:
    return {
        "topic_seed": seed.topic,
        "domain": seed.domain,
        "acm_id": seed.acm_id,
        "target_category": category.value,
        "seed_metadata": {
            "seed_id": seed.seed_id,
            "sample_index": seed.sample_index,
            "source_row_count": seed.source_row_count,
            "source_leaf_label": seed.source_leaf_label,
        },
    }


def _build_query_payload(
    seed: PasaTopicSeed,
    plan: PasaExpansionPlan,
    recent_openings: list[str],
) -> dict[str, object]:
    return {
        "topic_seed": seed.topic,
        "domain": seed.domain,
        "acm_id": seed.acm_id,
        "target_category": plan.category.value,
        "plan": cast(dict[str, object], plan.dict()),
        "recent_opening_examples": recent_openings,
    }


def _build_validation_payload(
    seed: PasaTopicSeed,
    category: PasaCategory,
    query: str,
    recent_openings: list[str],
) -> dict[str, object]:
    return {
        "topic_seed": seed.topic,
        "domain": seed.domain,
        "acm_id": seed.acm_id,
        "target_category": category.value,
        "query": query,
        "recent_opening_examples": recent_openings,
    }


def _build_rewrite_payload(
    seed: PasaTopicSeed,
    plan: PasaExpansionPlan,
    draft_query: str,
    validation: PasaValidationResponse,
    recent_openings: list[str],
) -> dict[str, object]:
    return {
        "topic_seed": seed.topic,
        "domain": seed.domain,
        "acm_id": seed.acm_id,
        "target_category": plan.category.value,
        "plan": cast(dict[str, object], plan.dict()),
        "draft_query": draft_query,
        "validation_result": cast(dict[str, object], validation.dict()),
        "recent_opening_examples": recent_openings,
    }


def _generate_single_record(
    *,
    client: OpenAICompatibleClient,
    prompts: PasaPromptConfig,
    seed: PasaTopicSeed,
    category: PasaCategory,
    query_id: str,
    trace_id: str,
    model_name: str,
    max_rewrite_attempts: int,
    recent_openings: list[str],
) -> tuple[PasaQueryRecord | None, PasaGenerationTrace]:
    trace = PasaGenerationTrace(
        trace_id=trace_id,
        query_id=query_id,
        seed_id=seed.seed_id,
        category=category,
        topic_seed=seed,
        final_status="started",
    )

    try:
        plan_payload = _build_plan_payload(seed, category)
        plan_prompt = _serialize_payload(plan_payload)
        trace.plan_prompt = plan_prompt
        plan = client.generate_structured(
            system_prompt=prompts.plan_generation,
            user_prompt=plan_prompt,
            response_model=PasaExpansionPlan,
        )
        trace.plan_response = plan

        query_payload = _build_query_payload(seed, plan, recent_openings)
        query_prompt = _serialize_payload(query_payload)
        trace.query_prompt = query_prompt
        draft = client.generate_structured(
            system_prompt=prompts.query_generation,
            user_prompt=query_prompt,
            response_model=PasaQueryDraftResponse,
        )
        trace.query_response = draft

        candidate_query = draft.query.strip()
        if not candidate_query:
            raise ValueError("Generated draft query is empty.")

        validation_payload = _build_validation_payload(seed, category, candidate_query, recent_openings)
        validation_prompt = _serialize_payload(validation_payload)
        trace.validation_prompt = validation_prompt
        validation = client.generate_structured(
            system_prompt=prompts.validation,
            user_prompt=validation_prompt,
            response_model=PasaValidationResponse,
        )
        validation = _override_validation_for_surface_quality(validation, candidate_query, recent_openings)
        trace.validation_response = validation

        final_query = candidate_query
        final_validation = validation
        had_rewrite = False

        if not validation.is_valid and validation.rewrite_needed and max_rewrite_attempts > 0:
            rewrite_payload = _build_rewrite_payload(
                seed,
                plan,
                candidate_query,
                validation,
                recent_openings,
            )
            rewrite_prompt = _serialize_payload(rewrite_payload)
            trace.rewrite_prompt = rewrite_prompt
            rewrite = client.generate_structured(
                system_prompt=prompts.rewrite,
                user_prompt=rewrite_prompt,
                response_model=PasaRewriteResponse,
            )
            trace.rewrite_response = rewrite

            rewritten_query = rewrite.query.strip()
            if rewritten_query:
                final_query = rewritten_query
                had_rewrite = True
                post_rewrite_validation_payload = _build_validation_payload(
                    seed,
                    category,
                    final_query,
                    recent_openings,
                )
                post_rewrite_validation_prompt = _serialize_payload(post_rewrite_validation_payload)
                trace.post_rewrite_validation_prompt = post_rewrite_validation_prompt
                final_validation = client.generate_structured(
                    system_prompt=prompts.validation,
                    user_prompt=post_rewrite_validation_prompt,
                    response_model=PasaValidationResponse,
                )
                final_validation = _override_validation_for_surface_quality(
                    final_validation,
                    final_query,
                    recent_openings,
                )
                trace.post_rewrite_validation_response = final_validation

        trace.final_query = final_query
        if not final_validation.is_valid:
            trace.final_status = "rejected"
            return None, trace

        record = PasaQueryRecord(
            query_id=query_id,
            trace_id=trace_id,
            source_type="topic_to_pasa_query",
            source_detail=seed.seed_id,
            topic_seed=seed,
            category=category,
            plan_constraint_kind=plan.constraint_kind,
            plan_constraint_value=plan.constraint_value,
            draft_query=draft.query.strip(),
            final_query=final_query,
            validation_passed=True,
            had_rewrite=had_rewrite,
            llm_model=model_name,
            llm_prompt_version=prompts.prompt_version,
        )
        trace.final_status = "accepted"
        return record, trace
    except Exception as exc:
        trace.final_status = "error"
        trace.error_message = str(exc)
        return None, trace


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _build_report(
    *,
    total_unique_seeds: int,
    selected_seed_count: int,
    output_records: list[PasaQueryRecord],
    run_status_counts: Counter[str],
    skipped_existing: int,
) -> dict[str, object]:
    category_counts: Counter[str] = Counter()
    rewritten_count = 0
    per_seed_counts: Counter[str] = Counter()

    for record in output_records:
        category_counts[record.category.value] += 1
        per_seed_counts[record.topic_seed.seed_id] += 1
        if record.had_rewrite:
            rewritten_count += 1

    complete_seed_count = sum(1 for count in per_seed_counts.values() if count == len(PasaCategory))

    return {
        "total_unique_seeds": total_unique_seeds,
        "selected_seed_count": selected_seed_count,
        "expected_queries_for_selection": selected_seed_count * len(PasaCategory),
        "output_query_count": len(output_records),
        "complete_seed_count": complete_seed_count,
        "rewritten_query_count": rewritten_count,
        "skipped_existing_count": skipped_existing,
        "category_counts": _sorted_counter(category_counts),
        "run_status_counts": _sorted_counter(run_status_counts),
    }


def main() -> None:
    args = parse_args()
    root_dir = args.root.resolve()
    config_path = args.config.resolve()
    config = _load_config(config_path)

    seed_path = (args.seed_file.resolve() if args.seed_file is not None else _resolve_path(root_dir, config.paths.seed_file))
    prompt_path = _resolve_path(root_dir, config.paths.prompt_file)
    output_path = (args.output_file.resolve() if args.output_file is not None else _resolve_path(root_dir, config.paths.output_file))
    trace_path = (args.trace_file.resolve() if args.trace_file is not None else _resolve_path(root_dir, config.paths.trace_file))
    report_path = (args.report_file.resolve() if args.report_file is not None else _resolve_path(root_dir, config.paths.report_file))
    cache_path = _resolve_path(root_dir, config.paths.cache_file)

    if not config.runtime.resume and (output_path.exists() or trace_path.exists()):
        raise ValueError("Output or trace file already exists while resume=false.")

    prompts = _load_prompts(prompt_path)
    client, model_name = _load_llm_client(root_dir=root_dir, config=config, cache_path=cache_path)

    all_seeds = build_topic_seeds(seed_path)
    selected_seeds = _slice_seeds(all_seeds, args.start, args.limit)
    existing_records = _load_existing_records(output_path)
    processed_pairs = _load_processed_pairs(existing_records) if config.runtime.resume else set()
    next_index = _next_record_index(existing_records)

    run_status_counts: Counter[str] = Counter()
    skipped_existing = 0
    processed_seed_counter = 0

    for seed in selected_seeds:
        processed_seed_counter += 1
        for category in PasaCategory:
            pair_key = (seed.seed_id, category)
            if pair_key in processed_pairs:
                skipped_existing += 1
                run_status_counts["skipped_existing"] += 1
                continue

            recent_openings = _collect_recent_openings(existing_records)
            query_id = _make_query_id(next_index)
            trace_id = _make_trace_id(next_index)
            record, trace = _generate_single_record(
                client=client,
                prompts=prompts,
                seed=seed,
                category=category,
                query_id=query_id,
                trace_id=trace_id,
                model_name=model_name,
                max_rewrite_attempts=config.runtime.max_rewrite_attempts,
                recent_openings=recent_openings,
            )
            _append_model_jsonl(trace_path, trace)
            if record is not None:
                _append_model_jsonl(output_path, record)
                existing_records.append(record)
                processed_pairs.add(pair_key)
            run_status_counts[trace.final_status] += 1
            next_index += 1

        if processed_seed_counter % 25 == 0:
            print(
                f"Processed {processed_seed_counter}/{len(selected_seeds)} seeds. "
                f"Accepted={run_status_counts['accepted']} "
                f"Rejected={run_status_counts['rejected']} "
                f"Errors={run_status_counts['error']} "
                f"Skipped={run_status_counts['skipped_existing']}"
            )

    report = _build_report(
        total_unique_seeds=len(all_seeds),
        selected_seed_count=len(selected_seeds),
        output_records=existing_records,
        run_status_counts=run_status_counts,
        skipped_existing=skipped_existing,
    )
    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Unique deduplicated seeds: {len(all_seeds)}")
    print(f"Selected seeds: {len(selected_seeds)}")
    print(f"Accepted queries: {run_status_counts['accepted']}")
    print(f"Rejected queries: {run_status_counts['rejected']}")
    print(f"Errored queries: {run_status_counts['error']}")
    print(f"Skipped existing queries: {run_status_counts['skipped_existing']}")
    print(f"Output JSONL: {output_path}")
    print(f"Trace JSONL: {trace_path}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
