from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    AdjudicationStatus,
    DedupVerdict,
    DifficultyLabel,
    HardFilterStatus,
    QueryType,
    RecallRequirement,
    Scope,
    SourceType,
    TerminologyExplicitness,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicSeed(BaseModel):
    topic_id: str
    topic: str
    domain: str
    target_query_type: QueryType | None = None
    difficulty_hint: str | None = None
    sample_count: int | None = None
    source_stratum: str | None = None
    source_acm_id: str | None = None
    source_leaf_label: str | None = None


class QueryRecord(BaseModel):
    query_id: str
    raw_query: str
    source_type: SourceType
    source_detail: str
    topic_seed: TopicSeed | None = None
    language: str = "en"
    domain: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    target_query_type: QueryType | None = None

    surface_form: str | None = None
    canonical_query: str | None = None
    topic_guess: str | None = None
    notes: str | None = None

    hard_filter_status: HardFilterStatus | None = None
    hard_filter_reason: str | None = None

    is_decidable: bool | None = None
    has_clear_boundary: bool | None = None
    is_non_degenerate: bool | None = None
    no_fulltext_dependency: bool | None = None
    type_consistent: bool | None = None
    type_consistency_reason: str | None = None
    quality_rationale: str | None = None
    review_needed: bool = False

    cluster_id: str | None = None
    duplicate_group_id: str | None = None
    is_cluster_representative: bool = True

    query_type: QueryType | None = None
    topics: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    analysis_targets: list[str] = Field(default_factory=list)
    scope: Scope | None = None
    difficulty: DifficultyLabel | None = None
    terminology_explicitness: TerminologyExplicitness | None = None
    recall_requirement: RecallRequirement | None = None

    adjudication_status: AdjudicationStatus | None = None
    adjudication_notes: str | None = None
    release_version: str | None = None

    llm_model: str | None = None
    llm_prompt_version: str | None = None
    pipeline_run_id: str | None = None

    def quality_pass_count(self) -> int:
        checks = [
            self.is_decidable,
            self.has_clear_boundary,
            self.is_non_degenerate,
            self.no_fulltext_dependency,
            self.type_consistent,
        ]
        return sum(1 for c in checks if c is True)


class QueryGenerationResponse(BaseModel):
    queries: list[str]


class NormalizationResponse(BaseModel):
    surface_form: str
    canonical_query: str
    topic_guess: str
    notes: str | None = None


class QualityCheckResponse(BaseModel):
    is_decidable: bool
    has_clear_boundary: bool
    is_non_degenerate: bool
    no_fulltext_dependency: bool
    rationale: str
    review_needed: bool = False


class ClassificationResponse(BaseModel):
    query_type: QueryType
    topics: list[str]
    methods: list[str]
    applications: list[str]
    analysis_targets: list[str]
    scope: Scope
    difficulty: DifficultyLabel
    terminology_explicitness: TerminologyExplicitness
    recall_requirement: RecallRequirement
    domain: str
    rationale: str


class DedupJudgeResponse(BaseModel):
    verdict: DedupVerdict
    confidence: float
    rationale: str


class AdjudicationResponse(BaseModel):
    adjudication_status: AdjudicationStatus
    rationale: str
    revision_suggestion: str | None = None


class PipelinePaths(BaseModel):
    topic_seeds: str
    raw_queries: str
    normalized_queries: str
    enriched_queries: str
    filtered_queries: str
    clustered_queries: str
    balanced_queries: str
    adjudicated_queries: str
    released_queries: str
    logs_dir: str
    llm_cache: str


class RuntimeConfig(BaseModel):
    random_seed: int = 42
    enable_llm_cache: bool = True
    max_llm_retries: int = 3
    llm_temperature: float = 0.2
    candidate_duplicate_k: int = 8
    duplicate_similarity_threshold: float = 0.84
    min_quality_pass_count: int = 4


class StageConfig(BaseModel):
    run_stage1_generate: bool = True
    run_stage2_normalize: bool = True
    run_stage3_enrich: bool = True
    run_stage4_filter: bool = True
    run_stage5_dedup: bool = True
    run_stage6_balance: bool = True
    run_stage7_adjudicate: bool = True
    run_stage8_release: bool = True


class PipelineConfig(BaseModel):
    paths: PipelinePaths
    runtime: RuntimeConfig
    stages: StageConfig


class SamplingConfig(BaseModel):
    release: dict[str, int]


class PromptConfig(BaseModel):
    query_generation: str
    rewrite: str
    blindspot: str
    normalize: str
    quality_check: str
    classify: str
    dedup_judge: str
    adjudicate: str


class GenerationConfig(BaseModel):
    per_seed_topic_queries: int
    per_seed_rewrites: int
    blindspot_queries_per_domain: int
    sample_count_scale: int = 1
    max_topic_queries_per_seed: int | None = None
    max_rewrites_per_seed: int | None = None
    blindspots: list[str]


class TaxonomyTypeConfig(BaseModel):
    description: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    generation_tips: list[str]
    label_fields: list[str]


class TaxonomyConstraints(BaseModel):
    word_count_min: int
    word_count_max: int
    banned_phrases: list[str]


class TaxonomyConfig(BaseModel):
    priority_order: list[QueryType]
    type_distribution: dict[str, float]
    constraints: TaxonomyConstraints
    types: dict[str, TaxonomyTypeConfig]


class JsonlRecord(BaseModel):
    payload: dict[str, Any]
