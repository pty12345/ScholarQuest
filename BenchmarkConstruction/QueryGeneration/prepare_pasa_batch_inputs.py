from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

QUERY_GENERATION_DIR = Path(__file__).resolve().parent
if str(QUERY_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUERY_GENERATION_DIR))

from pasa_types import PasaCategory, PasaGenerationConfig, PasaTopicSeed


DEFAULT_OUTPUT_DIR = QUERY_GENERATION_DIR / "output" / "pasa_batch_inputs"
DEFAULT_SHUFFLE_SEED = 20260426


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DashScope Batch API JSONL inputs for query generation.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=QUERY_GENERATION_DIR / "pasa_generation_config.yaml",
        help="Path to the generation config file.",
    )
    parser.add_argument("--seed-file", type=Path, default=None, help="Optional override for the input seed JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for batch input JSONL files.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Topic seeds per JSONL file.")
    parser.add_argument("--num-files", type=int, default=10, help="Number of JSONL files to write.")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED, help="Seed for deterministic shuffling.")
    parser.add_argument("--model", type=str, default=None, help="Optional model override.")
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, object], payload)


def _load_config(config_path: Path) -> PasaGenerationConfig:
    return PasaGenerationConfig(**_load_yaml_mapping(config_path))


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return ""


def _read_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _read_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_topic_seed_rows(seed_path: Path) -> list[PasaTopicSeed]:
    seeds: list[PasaTopicSeed] = []
    with seed_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at {seed_path}:{line_no}")

            topic = _normalize_text(payload.get("leaf_label"))
            domain = _normalize_text(payload.get("stratum"))
            acm_id = _normalize_text(payload.get("acm_id"))
            if not topic or not domain or not acm_id:
                continue

            seeds.append(
                PasaTopicSeed(
                    seed_id=f"PTS_{len(seeds) + 1:06d}",
                    topic=topic,
                    domain=domain,
                    acm_id=acm_id,
                    sample_index=_read_int(payload.get("sample_index")),
                    stratum_relative_weight=_read_float(payload.get("stratum_relative_weight")),
                    stratum_count_csv=_read_int(payload.get("stratum_count_csv")),
                    stratum_allocation=_read_int(payload.get("stratum_allocation")),
                    source_row_count=1,
                    source_stratum=domain,
                    source_leaf_label=topic,
                )
            )
    return seeds


def _system_prompt() -> str:
    categories = "\n".join(f"- {category.value}" for category in PasaCategory)
    return f"""You generate paper retrieval queries from one topic seed.

Return strict JSON only. The response must be one valid JSON object with this schema:
{{
  "seed_id": "PTS_000001",
  "topic_seed": "large language model agents",
  "domain": "cs.AI",
  "queries": [
    {{
      "category": "method_capability",
      "constraint_kind": "training_paradigm",
      "constraint_value": "reinforcement learning",
      "query": "Which papers study large language model agents trained with reinforcement learning?",
      "rationale": "reinforcement learning is a searchable technical constraint",
      "risk_flags": []
    }}
  ]
}}

Generate exactly one query for each query category:
{categories}

Category examples:
- method_capability:
  {{
    "category": "method_capability",
    "constraint_kind": "training_paradigm",
    "constraint_value": "reinforcement learning",
    "query": "Which papers study large language model agents trained with reinforcement learning?",
    "rationale": "the training paradigm is a concrete method constraint",
    "risk_flags": []
  }}
- setting_anchor:
  {{
    "category": "setting_anchor",
    "constraint_kind": "benchmark",
    "constraint_value": "HotpotQA",
    "query": "Which papers evaluate large language model agents on the HotpotQA benchmark?",
    "rationale": "the benchmark anchors the retrieval setting",
    "risk_flags": []
  }}
- claim_comparison:
  {{
    "category": "claim_comparison",
    "constraint_kind": "comparison",
    "constraint_value": "outperforms single-agent prompting",
    "query": "Which papers report that multi-agent language model systems outperform single-agent prompting?",
    "rationale": "the comparison gives a searchable claim direction",
    "risk_flags": []
  }}
- scope_control:
  {{
    "category": "scope_control",
    "constraint_kind": "exclude_filtering",
    "constraint_value": "tool-use agents",
    "query": "I am looking for papers on large language model agents that exclude tool-use agents from their scope.",
    "rationale": "the exclusion constraint narrows the retrieval scope",
    "risk_flags": []
  }}

Rules:
- Produce exactly four query objects, one per category.
- Each query must be a paper retrieval request.
- Keep each query concise, natural, and one sentence.
- Include one strong executable retrieval constraint per query.
- Stay faithful to the topic seed, domain, and ACM id.
- Avoid temporal wording such as recent, latest, since 2020, after 2020, or similar time constraints.
- Do not ask for analysis, advice, or long-form synthesis.
- Do not use fragment wording such as "Papers on ...".
- Vary the surface form across the four queries when natural."""


def _user_payload(seed: PasaTopicSeed) -> dict[str, Any]:
    return {
        "topic_seed": seed.topic,
        "domain": seed.domain,
        "acm_id": seed.acm_id,
        "seed_metadata": {
            "seed_id": seed.seed_id,
            "sample_index": seed.sample_index,
            "source_row_count": seed.source_row_count,
            "source_leaf_label": seed.source_leaf_label,
        },
        "target_categories": [category.value for category in PasaCategory],
    }


def _batch_request(seed: PasaTopicSeed, *, model: str, temperature: float) -> dict[str, Any]:
    return {
        "custom_id": f"query-{seed.seed_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": json.dumps(_user_payload(seed), ensure_ascii=True, sort_keys=True)},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.num_files <= 0:
        raise ValueError("--num-files must be positive")

    root_dir = args.root.resolve()
    config = _load_config(args.config.resolve())
    seed_path = args.seed_file.resolve() if args.seed_file is not None else _resolve_path(root_dir, config.paths.seed_file)
    output_dir = args.output_dir.resolve()
    model = args.model or config.llm.model

    seeds = _load_topic_seed_rows(seed_path)
    required_seed_count = args.chunk_size * args.num_files
    if len(seeds) < required_seed_count:
        raise ValueError(f"Need {required_seed_count} topic seeds but only found {len(seeds)}.")

    shuffled_seeds = list(seeds)
    random.Random(args.shuffle_seed).shuffle(shuffled_seeds)
    selected_seeds = shuffled_seeds[:required_seed_count]

    manifest: dict[str, Any] = {
        "seed_file": str(seed_path),
        "output_dir": str(output_dir),
        "shuffle_seed": args.shuffle_seed,
        "model": model,
        "temperature": config.llm.temperature,
        "chunk_size": args.chunk_size,
        "num_files": args.num_files,
        "total_selected_topic_seeds": len(selected_seeds),
        "files": [],
    }

    for file_index in range(args.num_files):
        start = file_index * args.chunk_size
        end = start + args.chunk_size
        chunk = selected_seeds[start:end]
        file_name = f"pasa_qwen_batch_input_{file_index + 1:02d}.jsonl"
        output_path = output_dir / file_name
        rows = [_batch_request(seed, model=model, temperature=config.llm.temperature) for seed in chunk]
        _write_jsonl(output_path, rows)
        manifest["files"].append(
            {
                "path": str(output_path),
                "topic_seed_count": len(chunk),
                "custom_id_start": rows[0]["custom_id"],
                "custom_id_end": rows[-1]["custom_id"],
            }
        )

    manifest_path = output_dir / "pasa_qwen_batch_manifest.json"
    _ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {args.num_files} batch input files to {output_dir}")
    print(f"Total topic seeds: {len(selected_seeds)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
