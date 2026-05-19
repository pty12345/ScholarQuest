from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from ..io_utils import append_jsonl, read_jsonl, write_jsonl
from ..services.pipeline_services import (
    AdjudicationService,
    BalancingService,
    DedupService,
    EnrichmentService,
    FilteringService,
    GenerationService,
    NormalizationService,
)
from ..types import QueryRecord, TopicSeed


@dataclass(slots=True)
class StagePaths:
    topic_seeds: Path
    raw_queries: Path
    normalized_queries: Path
    enriched_queries: Path
    filtered_queries: Path
    clustered_queries: Path
    balanced_queries: Path
    adjudicated_queries: Path
    released_queries: Path
    logs_dir: Path


class StageRunner:
    def __init__(
        self,
        *,
        paths: StagePaths,
        generation_service: GenerationService,
        normalization_service: NormalizationService,
        filtering_service: FilteringService,
        dedup_service: DedupService,
        enrichment_service: EnrichmentService,
        balancing_service: BalancingService,
        adjudication_service: AdjudicationService,
        per_seed_topic_queries: int,
        per_seed_rewrites: int,
        blindspot_queries_per_domain: int,
        target_size: int,
    ) -> None:
        self._paths = paths
        self._generation_service = generation_service
        self._normalization_service = normalization_service
        self._filtering_service = filtering_service
        self._dedup_service = dedup_service
        self._enrichment_service = enrichment_service
        self._balancing_service = balancing_service
        self._adjudication_service = adjudication_service
        self._per_seed_topic_queries = per_seed_topic_queries
        self._per_seed_rewrites = per_seed_rewrites
        self._blindspot_queries_per_domain = blindspot_queries_per_domain
        self._target_size = target_size

    def _log_stage(self, stage_name: str, records: list[QueryRecord]) -> None:
        accepted = sum(1 for record in records if record.hard_filter_status is None or record.hard_filter_status.value != "reject")
        review_needed = sum(1 for record in records if record.review_needed)
        adjudicated = sum(1 for record in records if record.adjudication_status is not None)
        append_jsonl(
            self._paths.logs_dir / "pipeline_runs.jsonl",
            {
                "stage": stage_name,
                "total_records": len(records),
                "accepted_records": accepted,
                "review_needed_records": review_needed,
                "adjudicated_records": adjudicated,
            },
        )

    def stage1_generate(self, seeds: list[TopicSeed] | None = None) -> list[QueryRecord]:
        source_seeds = seeds or read_jsonl(self._paths.topic_seeds, TopicSeed)
        write_jsonl(self._paths.topic_seeds, source_seeds)
        records = self._generation_service.generate(
            source_seeds,
            per_seed_topic_queries=self._per_seed_topic_queries,
            per_seed_rewrites=self._per_seed_rewrites,
            blindspot_queries_per_domain=self._blindspot_queries_per_domain,
        )
        write_jsonl(self._paths.raw_queries, records)
        self._log_stage("stage1_generate", records)
        return records

    def stage2_normalize(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.raw_queries, QueryRecord)
        normalized = self._normalization_service.normalize(source)
        write_jsonl(self._paths.normalized_queries, normalized)
        self._log_stage("stage2_normalize", normalized)
        return normalized

    def stage3_enrich(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.normalized_queries, QueryRecord)
        enriched = self._enrichment_service.enrich(source)
        write_jsonl(self._paths.enriched_queries, enriched)
        self._log_stage("stage3_enrich", enriched)
        return enriched

    def stage4_filter(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.enriched_queries, QueryRecord)
        filtered = self._filtering_service.filter(source)
        write_jsonl(self._paths.filtered_queries, filtered)
        self._log_stage("stage4_filter", filtered)
        return filtered

    def stage5_dedup(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.filtered_queries, QueryRecord)
        clustered = self._dedup_service.deduplicate(source)
        write_jsonl(self._paths.clustered_queries, clustered)
        self._log_stage("stage5_dedup", clustered)
        return clustered

    def stage6_balance(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.clustered_queries, QueryRecord)
        balanced = self._balancing_service.balance(source, self._target_size)
        write_jsonl(self._paths.balanced_queries, balanced)
        self._log_stage("stage6_balance", balanced)
        return balanced

    def stage7_adjudicate(self, records: list[QueryRecord] | None = None) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.balanced_queries, QueryRecord)
        adjudicated = self._adjudication_service.adjudicate(source)
        write_jsonl(self._paths.adjudicated_queries, adjudicated)
        self._log_stage("stage7_adjudicate", adjudicated)
        return adjudicated

    def stage8_release(
        self,
        records: list[QueryRecord] | None = None,
        *,
        release_version: str,
    ) -> list[QueryRecord]:
        source = records or read_jsonl(self._paths.adjudicated_queries, QueryRecord)
        released = []
        for record in source:
            if record.adjudication_status is None or record.adjudication_status.value != "accept":
                continue
            updated = copy.deepcopy(record)
            updated.release_version = release_version
            released.append(updated)
        write_jsonl(self._paths.released_queries, released)
        self._log_stage("stage8_release", released)
        return released
