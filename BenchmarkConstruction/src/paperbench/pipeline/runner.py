from __future__ import annotations

from pathlib import Path

from ..agents.adjudication_agent import AdjudicationAgent
from ..agents.classification_agent import ClassificationAgent
from ..agents.dedup_judge_agent import DedupJudgeAgent
from ..agents.normalization_agent import NormalizationAgent
from ..agents.query_generation_agent import QueryGenerationAgent
from ..agents.quality_scoring_agent import QualityCheckAgent
from ..embeddings.local_embedder import LocalEmbedder
from ..llm.client import OpenAICompatibleClient
from ..services.pipeline_services import (
    AdjudicationService,
    BalancingService,
    DedupService,
    EnrichmentService,
    FilteringService,
    GenerationService,
    NormalizationService,
)
from ..settings import AppConfig
from ..types import TopicSeed
from .stages import StagePaths, StageRunner


class QueryPipelineRunner:
    def __init__(self, config: AppConfig) -> None:
        cache_path = config.root_dir / config.pipeline.paths.llm_cache
        client = OpenAICompatibleClient(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            cache_path=cache_path if config.pipeline.runtime.enable_llm_cache else None,
            temperature=config.pipeline.runtime.llm_temperature,
        )

        generation_service = GenerationService(
            generation_agent=QueryGenerationAgent(client, config.prompts.query_generation),
            rewrite_agent=QueryGenerationAgent(client, config.prompts.rewrite),
            blindspot_agent=QueryGenerationAgent(client, config.prompts.blindspot),
            blindspots=config.generation.blindspots,
            random_seed=config.pipeline.runtime.random_seed,
            taxonomy=config.taxonomy,
            sample_count_scale=config.generation.sample_count_scale,
            max_topic_queries_per_seed=config.generation.max_topic_queries_per_seed,
            max_rewrites_per_seed=config.generation.max_rewrites_per_seed,
        )
        normalization_service = NormalizationService(
            NormalizationAgent(client, config.prompts.normalize)
        )
        filtering_service = FilteringService(
            QualityCheckAgent(client, config.prompts.quality_check),
            min_quality_pass_count=config.pipeline.runtime.min_quality_pass_count,
        )
        dedup_service = DedupService(
            embedder=LocalEmbedder(config.llm.local_embedding_model),
            judge_agent=DedupJudgeAgent(client, config.prompts.dedup_judge),
            candidate_k=config.pipeline.runtime.candidate_duplicate_k,
            similarity_threshold=config.pipeline.runtime.duplicate_similarity_threshold,
        )
        enrichment_service = EnrichmentService(
            ClassificationAgent(client, config.prompts.classify)
        )
        balancing_service = BalancingService(
            type_distribution=config.taxonomy.type_distribution,
        )
        adjudication_service = AdjudicationService(
            AdjudicationAgent(client, config.prompts.adjudicate)
        )

        paths = StagePaths(
            topic_seeds=config.root_dir / config.pipeline.paths.topic_seeds,
            raw_queries=config.root_dir / config.pipeline.paths.raw_queries,
            normalized_queries=config.root_dir / config.pipeline.paths.normalized_queries,
            enriched_queries=config.root_dir / config.pipeline.paths.enriched_queries,
            filtered_queries=config.root_dir / config.pipeline.paths.filtered_queries,
            clustered_queries=config.root_dir / config.pipeline.paths.clustered_queries,
            balanced_queries=config.root_dir / config.pipeline.paths.balanced_queries,
            adjudicated_queries=config.root_dir / config.pipeline.paths.adjudicated_queries,
            released_queries=config.root_dir / config.pipeline.paths.released_queries,
            logs_dir=config.root_dir / config.pipeline.paths.logs_dir,
        )

        self._stages = StageRunner(
            paths=paths,
            generation_service=generation_service,
            normalization_service=normalization_service,
            filtering_service=filtering_service,
            dedup_service=dedup_service,
            enrichment_service=enrichment_service,
            balancing_service=balancing_service,
            adjudication_service=adjudication_service,
            per_seed_topic_queries=config.generation.per_seed_topic_queries,
            per_seed_rewrites=config.generation.per_seed_rewrites,
            blindspot_queries_per_domain=config.generation.blindspot_queries_per_domain,
            target_size=config.sampling.release["target_size"],
        )
        self._config = config

    @classmethod
    def from_root(cls, root_dir: Path | None = None) -> "QueryPipelineRunner":
        from ..settings import load_app_config

        return cls(load_app_config(root_dir))

    def run(self, seeds: list[TopicSeed] | None = None) -> None:
        records = None
        stages = self._config.pipeline.stages
        if stages.run_stage1_generate:
            records = self._stages.stage1_generate(seeds)
        if stages.run_stage2_normalize:
            records = self._stages.stage2_normalize(records)
        if stages.run_stage3_enrich:
            records = self._stages.stage3_enrich(records)
        if stages.run_stage4_filter:
            records = self._stages.stage4_filter(records)
        if stages.run_stage5_dedup:
            records = self._stages.stage5_dedup(records)
        if stages.run_stage6_balance:
            records = self._stages.stage6_balance(records)
        if stages.run_stage7_adjudicate:
            records = self._stages.stage7_adjudicate(records)
        if stages.run_stage8_release and records is not None:
            self._stages.stage8_release(records, release_version="v0.1")
