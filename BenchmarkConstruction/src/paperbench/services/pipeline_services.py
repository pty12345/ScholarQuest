from __future__ import annotations

import copy
import random
from collections import defaultdict

from ..agents.adjudication_agent import AdjudicationAgent
from ..agents.classification_agent import ClassificationAgent
from ..agents.dedup_judge_agent import DedupJudgeAgent
from ..agents.normalization_agent import NormalizationAgent
from ..agents.query_generation_agent import QueryGenerationAgent
from ..agents.quality_scoring_agent import QualityCheckAgent
from ..embeddings.local_embedder import LocalEmbedder, cosine_similarity
from ..enums import DedupVerdict, HardFilterStatus, SourceType
from ..ids import make_cluster_id, make_duplicate_group_id, make_query_id
from ..rules.cleaning import clean_surface_query
from ..rules.hard_filters import run_hard_filter
from ..rules.sampling import stratified_sample
from ..rules.taxonomy import evaluate_type_consistency
from ..types import QueryRecord, TaxonomyConfig, TopicSeed


class GenerationService:
    def __init__(
        self,
        *,
        generation_agent: QueryGenerationAgent,
        rewrite_agent: QueryGenerationAgent,
        blindspot_agent: QueryGenerationAgent,
        blindspots: list[str],
        random_seed: int,
        taxonomy: TaxonomyConfig,
        sample_count_scale: int,
        max_topic_queries_per_seed: int | None,
        max_rewrites_per_seed: int | None,
    ) -> None:
        self._generation_agent = generation_agent
        self._rewrite_agent = rewrite_agent
        self._blindspot_agent = blindspot_agent
        self._blindspots = blindspots
        self._rng = random.Random(random_seed)
        self._taxonomy = taxonomy
        self._sample_count_scale = max(1, sample_count_scale)
        self._max_topic_queries_per_seed = max_topic_queries_per_seed
        self._max_rewrites_per_seed = max_rewrites_per_seed

    def _type_definition(self, target_query_type: str) -> tuple[str, list[str]]:
        type_config = self._taxonomy.types[target_query_type]
        return type_config.description, type_config.generation_tips

    def _scaled_count(self, base_count: int, sample_count: int | None, max_count: int | None) -> int:
        if base_count <= 0:
            return 0
        if sample_count is None or sample_count <= 1:
            return base_count if max_count is None else min(base_count, max_count)

        extra = (sample_count - 1) // self._sample_count_scale
        scaled = base_count + extra
        if max_count is not None:
            scaled = min(scaled, max_count)
        return max(1, scaled)

    def _build_query_record(
        self,
        *,
        query_id: str,
        raw_query: str,
        source_type: SourceType,
        source_detail: str,
        seed: TopicSeed,
    ) -> QueryRecord:
        return QueryRecord(
            query_id=query_id,
            raw_query=raw_query,
            source_type=source_type,
            source_detail=source_detail,
            topic_seed=copy.deepcopy(seed),
            domain=seed.domain,
            target_query_type=seed.target_query_type,
        )

    def generate(
        self,
        seeds: list[TopicSeed],
        *,
        per_seed_topic_queries: int,
        per_seed_rewrites: int,
        blindspot_queries_per_domain: int,
    ) -> list[QueryRecord]:
        records: list[QueryRecord] = []
        current_index = 1

        for seed in seeds:
            if seed.target_query_type is None:
                continue
            type_definition, generation_tips = self._type_definition(seed.target_query_type.value)
            topic_query_count = self._scaled_count(
                per_seed_topic_queries,
                seed.sample_count,
                self._max_topic_queries_per_seed,
            )
            base_queries = self._generation_agent.generate_from_seed(
                seed_topic=seed.topic,
                domain=seed.domain,
                target_query_type=seed.target_query_type,
                difficulty_hint=seed.difficulty_hint,
                num_queries=topic_query_count,
                type_definition=type_definition,
                generation_tips=generation_tips,
            )
            for query in base_queries:
                records.append(
                    self._build_query_record(
                        query_id=make_query_id(current_index),
                        raw_query=query,
                        source_type=SourceType.TOPIC_TO_QUERY,
                        source_detail=seed.topic_id,
                        seed=seed,
                    )
                )
                current_index += 1

            rewrite_count = self._scaled_count(
                per_seed_rewrites,
                seed.sample_count,
                self._max_rewrites_per_seed,
            )
            sampled_for_rewrite = base_queries[:rewrite_count]
            for query in sampled_for_rewrite:
                rewrites = self._rewrite_agent.rewrite_query(
                    source_query=query,
                    domain=seed.domain,
                    target_query_type=seed.target_query_type,
                    difficulty_hint=seed.difficulty_hint,
                    num_queries=1,
                    type_definition=type_definition,
                    generation_tips=generation_tips,
                )
                for rewritten in rewrites:
                    records.append(
                        self._build_query_record(
                            query_id=make_query_id(current_index),
                            raw_query=rewritten,
                            source_type=SourceType.DIFFICULTY_REWRITE,
                            source_detail=seed.topic_id,
                            seed=seed,
                        )
                    )
                    current_index += 1

        domains = sorted({seed.domain for seed in seeds})
        for domain in domains:
            for blindspot in self._blindspots:
                for query_type in self._taxonomy.priority_order:
                    blindspot_seed = TopicSeed(
                        topic_id=f"BLIND_{domain}_{blindspot}",
                        topic=f"{domain} {blindspot}",
                        domain=domain,
                        target_query_type=query_type,
                    )
                    type_definition, generation_tips = self._type_definition(query_type.value)
                    blindspot_queries = self._blindspot_agent.generate_blindspot(
                        blindspot=blindspot,
                        domain=domain,
                        target_query_type=query_type,
                        num_queries=blindspot_queries_per_domain,
                        type_definition=type_definition,
                        generation_tips=generation_tips,
                    )
                    for query in blindspot_queries:
                        records.append(
                            self._build_query_record(
                                query_id=make_query_id(current_index),
                                raw_query=query,
                                source_type=SourceType.BLINDSPOT_SUPPLEMENT,
                                source_detail=blindspot,
                                seed=blindspot_seed,
                            )
                        )
                        current_index += 1

        self._rng.shuffle(records)
        return records


class NormalizationService:
    def __init__(self, agent: NormalizationAgent) -> None:
        self._agent = agent

    def normalize(self, records: list[QueryRecord]) -> list[QueryRecord]:
        normalized: list[QueryRecord] = []
        for record in records:
            cleaned = clean_surface_query(record.raw_query)
            result = self._agent.normalize(cleaned)

            updated = copy.deepcopy(record)
            updated.surface_form = result.surface_form or cleaned
            updated.canonical_query = result.canonical_query
            updated.topic_guess = result.topic_guess
            updated.notes = result.notes
            normalized.append(updated)
        return normalized


class FilteringService:
    def __init__(self, agent: QualityCheckAgent, min_quality_pass_count: int) -> None:
        self._agent = agent
        self._min_quality_pass_count = min_quality_pass_count

    def filter(self, records: list[QueryRecord]) -> list[QueryRecord]:
        filtered: list[QueryRecord] = []
        for record in records:
            updated = copy.deepcopy(record)
            status, reason = run_hard_filter(updated.surface_form or updated.raw_query)
            updated.hard_filter_status = status
            updated.hard_filter_reason = reason
            updated.type_consistent, updated.type_consistency_reason = evaluate_type_consistency(updated)
            if status == HardFilterStatus.REJECT:
                filtered.append(updated)
                continue

            check = self._agent.check(updated)
            updated.is_decidable = check.is_decidable
            updated.has_clear_boundary = check.has_clear_boundary
            updated.is_non_degenerate = check.is_non_degenerate
            updated.no_fulltext_dependency = check.no_fulltext_dependency
            updated.quality_rationale = check.rationale
            updated.review_needed = check.review_needed or updated.quality_pass_count() < self._min_quality_pass_count
            filtered.append(updated)
        return filtered


class DedupService:
    def __init__(
        self,
        *,
        embedder: LocalEmbedder,
        judge_agent: DedupJudgeAgent,
        candidate_k: int,
        similarity_threshold: float,
    ) -> None:
        self._embedder = embedder
        self._judge_agent = judge_agent
        self._candidate_k = candidate_k
        self._similarity_threshold = similarity_threshold

    def deduplicate(self, records: list[QueryRecord]) -> list[QueryRecord]:
        active_records = [record for record in records if record.hard_filter_status != HardFilterStatus.REJECT]
        if not active_records:
            return records

        vectors = self._embedder.encode([record.canonical_query or record.raw_query for record in active_records])
        clusters: dict[int, list[int]] = defaultdict(list)
        cluster_index = 0

        for idx, vector in enumerate(vectors):
            assigned = False
            for existing_cluster_idx, member_indices in clusters.items():
                anchor_vector = vectors[member_indices[0]]
                if cosine_similarity(vector, anchor_vector) >= self._similarity_threshold:
                    member_indices.append(idx)
                    assigned = True
                    break
            if not assigned:
                clusters[cluster_index].append(idx)
                cluster_index += 1

        updated_records = {record.query_id: copy.deepcopy(record) for record in records}
        duplicate_group_count = 1

        for cluster_idx, member_indices in clusters.items():
            cluster_id = make_cluster_id(cluster_idx + 1)
            representative_idx = member_indices[0]
            representative = active_records[representative_idx]
            updated_records[representative.query_id].cluster_id = cluster_id
            updated_records[representative.query_id].is_cluster_representative = True

            neighbor_indices = member_indices[: self._candidate_k]
            for member_idx in neighbor_indices[1:]:
                candidate = active_records[member_idx]
                updated_records[candidate.query_id].cluster_id = cluster_id
                judge = self._judge_agent.judge(representative, candidate)
                if judge.verdict == DedupVerdict.DUPLICATE:
                    duplicate_group_id = make_duplicate_group_id(duplicate_group_count)
                    updated_records[candidate.query_id].duplicate_group_id = duplicate_group_id
                    updated_records[candidate.query_id].is_cluster_representative = False
                    updated_records[representative.query_id].duplicate_group_id = duplicate_group_id
                    duplicate_group_count += 1
                else:
                    updated_records[candidate.query_id].is_cluster_representative = True

        return list(updated_records.values())


class EnrichmentService:
    def __init__(self, agent: ClassificationAgent) -> None:
        self._agent = agent

    def enrich(self, records: list[QueryRecord]) -> list[QueryRecord]:
        enriched: list[QueryRecord] = []
        for record in records:
            updated = copy.deepcopy(record)
            if updated.hard_filter_status == HardFilterStatus.REJECT:
                enriched.append(updated)
                continue
            classification = self._agent.classify(updated)
            updated.query_type = classification.query_type
            updated.topics = classification.topics
            updated.methods = classification.methods
            updated.applications = classification.applications
            updated.analysis_targets = classification.analysis_targets
            updated.scope = classification.scope
            updated.difficulty = classification.difficulty
            updated.terminology_explicitness = classification.terminology_explicitness
            updated.recall_requirement = classification.recall_requirement
            updated.domain = classification.domain or updated.domain
            updated.type_consistent, updated.type_consistency_reason = evaluate_type_consistency(updated)
            updated.notes = "\n".join(filter(None, [updated.notes, classification.rationale]))
            enriched.append(updated)
        return enriched


class BalancingService:
    def __init__(self, type_distribution: dict[str, float]) -> None:
        self._type_distribution = type_distribution

    def balance(self, records: list[QueryRecord], target_size: int) -> list[QueryRecord]:
        candidates = [
            record
            for record in records
            if record.hard_filter_status != HardFilterStatus.REJECT
            and record.is_cluster_representative
            and record.type_consistent is not False
            and not record.review_needed
        ]
        return stratified_sample(
            candidates,
            target_size=target_size,
            type_distribution=self._type_distribution,
        )


class AdjudicationService:
    def __init__(self, agent: AdjudicationAgent) -> None:
        self._agent = agent

    def adjudicate(self, records: list[QueryRecord]) -> list[QueryRecord]:
        decided: list[QueryRecord] = []
        for record in records:
            updated = copy.deepcopy(record)
            response = self._agent.adjudicate(updated)
            updated.adjudication_status = response.adjudication_status
            updated.adjudication_notes = response.rationale
            if response.revision_suggestion:
                updated.notes = "\n".join(filter(None, [updated.notes, response.revision_suggestion]))
            decided.append(updated)
        return decided
