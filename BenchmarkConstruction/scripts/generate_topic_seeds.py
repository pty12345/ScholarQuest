from __future__ import annotations

import argparse
from pathlib import Path

import _add_src_to_path  # noqa: F401

from paperbench.enums import QueryType
from paperbench.io_utils import write_jsonl
from paperbench.types import TopicSeed


DEFAULT_SEEDS = [
    # specific_topic seeds
    TopicSeed(
        topic_id="TS_0001",
        topic="long video description generation",
        domain="cv",
        target_query_type=QueryType.SPECIFIC_TOPIC,
        difficulty_hint="easy",
    ),
    TopicSeed(
        topic_id="TS_0002",
        topic="trigger-free document-level event extraction",
        domain="nlp",
        target_query_type=QueryType.SPECIFIC_TOPIC,
        difficulty_hint="easy",
    ),
    # cross_topic seeds
    TopicSeed(
        topic_id="TS_0003",
        topic="reinforcement learning for LLM agents",
        domain="agents_memory_tool_use",
        target_query_type=QueryType.CROSS_TOPIC,
        difficulty_hint="medium",
    ),
    TopicSeed(
        topic_id="TS_0004",
        topic="diffusion models for video generation",
        domain="cv",
        target_query_type=QueryType.CROSS_TOPIC,
        difficulty_hint="medium",
    ),
    TopicSeed(
        topic_id="TS_0005",
        topic="LLMs for stock factor mining",
        domain="nlp",
        target_query_type=QueryType.CROSS_TOPIC,
        difficulty_hint="medium",
    ),
    # method_architecture seeds
    TopicSeed(
        topic_id="TS_0006",
        topic="DPO for vision-language models",
        domain="multimodal",
        target_query_type=QueryType.METHOD_ARCHITECTURE,
        difficulty_hint="medium",
    ),
    TopicSeed(
        topic_id="TS_0007",
        topic="visual LLMs with MoE architecture",
        domain="multimodal",
        target_query_type=QueryType.METHOD_ARCHITECTURE,
        difficulty_hint="medium",
    ),
    TopicSeed(
        topic_id="TS_0008",
        topic="RLHF for hallucination reduction",
        domain="nlp",
        target_query_type=QueryType.METHOD_ARCHITECTURE,
        difficulty_hint="medium",
    ),
    # capability_application seeds
    TopicSeed(
        topic_id="TS_0009",
        topic="models that automatically write surveys from multiple papers",
        domain="nlp",
        target_query_type=QueryType.CAPABILITY_APPLICATION,
        difficulty_hint="hard",
    ),
    TopicSeed(
        topic_id="TS_0010",
        topic="VLM agents that can play PC games",
        domain="agents_memory_tool_use",
        target_query_type=QueryType.CAPABILITY_APPLICATION,
        difficulty_hint="hard",
    ),
    # collection_scoping seeds
    TopicSeed(
        topic_id="TS_0011",
        topic="all papers on machine translation agents",
        domain="nlp",
        target_query_type=QueryType.COLLECTION_SCOPING,
        difficulty_hint="medium",
    ),
    TopicSeed(
        topic_id="TS_0012",
        topic="datasets and benchmarks for robot task planning",
        domain="robotics",
        target_query_type=QueryType.COLLECTION_SCOPING,
        difficulty_hint="medium",
    ),
    # analysis_claim seeds
    TopicSeed(
        topic_id="TS_0013",
        topic="research on how in-context learning emerges during pretraining",
        domain="nlp",
        target_query_type=QueryType.ANALYSIS_CLAIM,
        difficulty_hint="hard",
    ),
    TopicSeed(
        topic_id="TS_0014",
        topic="papers on scaling laws for multimodal models",
        domain="multimodal",
        target_query_type=QueryType.ANALYSIS_CLAIM,
        difficulty_hint="hard",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write default topic seeds to a JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/seeds/topic_seeds.jsonl"),
        help="Output path for the generated seed file.",
    )
    args = parser.parse_args()
    write_jsonl(args.output, DEFAULT_SEEDS)
    print(f"Wrote {len(DEFAULT_SEEDS)} topic seeds to {args.output}")


if __name__ == "__main__":
    main()
