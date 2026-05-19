from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PasaCategory(str, Enum):
    METHOD_CAPABILITY = "method_capability"
    SETTING_ANCHOR = "setting_anchor"
    CLAIM_COMPARISON = "claim_comparison"
    SCOPE_CONTROL = "scope_control"


class PasaTopicSeed(BaseModel):
    seed_id: str
    topic: str
    domain: str
    acm_id: str
    sample_index: int | None = None
    stratum_relative_weight: float | None = None
    stratum_count_csv: int | None = None
    stratum_allocation: int | None = None
    source_row_count: int = 1
    source_stratum: str | None = None
    source_leaf_label: str | None = None


class PasaExpansionPlan(BaseModel):
    category: PasaCategory
    topic_seed: str
    constraint_kind: str
    constraint_value: str
    retrieval_intent: str
    rationale: str
    risk_flags: list[str] = Field(default_factory=list)


class PasaQueryDraftResponse(BaseModel):
    query: str


class PasaRewriteResponse(BaseModel):
    query: str
    revision_note: str


class PasaValidationResponse(BaseModel):
    is_valid: bool
    category_match: bool
    retrieval_oriented: bool
    executable_constraint: bool
    natural_language_ok: bool
    issues: list[str] = Field(default_factory=list)
    rewrite_needed: bool = False
    rewrite_instruction: str = ""
    rationale: str


class PasaQueryRecord(BaseModel):
    query_id: str
    trace_id: str
    source_type: str
    source_detail: str
    topic_seed: PasaTopicSeed
    category: PasaCategory
    plan_constraint_kind: str
    plan_constraint_value: str
    draft_query: str
    final_query: str
    validation_passed: bool
    had_rewrite: bool
    language: str = "en"
    generation_status: str = "accepted"
    llm_model: str
    llm_prompt_version: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class PasaGenerationTrace(BaseModel):
    trace_id: str
    query_id: str
    seed_id: str
    category: PasaCategory
    topic_seed: PasaTopicSeed
    plan_prompt: str | None = None
    plan_response: PasaExpansionPlan | None = None
    query_prompt: str | None = None
    query_response: PasaQueryDraftResponse | None = None
    validation_prompt: str | None = None
    validation_response: PasaValidationResponse | None = None
    rewrite_prompt: str | None = None
    rewrite_response: PasaRewriteResponse | None = None
    post_rewrite_validation_prompt: str | None = None
    post_rewrite_validation_response: PasaValidationResponse | None = None
    final_query: str | None = None
    final_status: str
    error_message: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class PasaPromptConfig(BaseModel):
    prompt_version: str
    plan_generation: str
    query_generation: str
    validation: str
    rewrite: str


class PasaPathConfig(BaseModel):
    seed_file: str
    prompt_file: str
    output_file: str
    trace_file: str
    report_file: str
    cache_file: str


class PasaLlmConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    batch_base_url_env: str = "OPENAI_BATCH_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    mode: str = "realtime"
    batch_base_url: str = "https://batch.dashscope.aliyuncs.com/compatible-mode/v1"


class PasaRuntimeConfig(BaseModel):
    max_rewrite_attempts: int = 1
    resume: bool = True


class PasaGenerationConfig(BaseModel):
    paths: PasaPathConfig
    llm: PasaLlmConfig = Field(default_factory=PasaLlmConfig)
    runtime: PasaRuntimeConfig = Field(default_factory=PasaRuntimeConfig)
