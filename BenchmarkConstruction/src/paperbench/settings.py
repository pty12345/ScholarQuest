from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from .types import (
    GenerationConfig,
    PipelineConfig,
    PromptConfig,
    SamplingConfig,
    TaxonomyConfig,
)


class LLMSettings(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"


class AppConfig(BaseModel):
    root_dir: Path
    llm: LLMSettings
    pipeline: PipelineConfig
    prompts: PromptConfig
    generation: GenerationConfig
    sampling: SamplingConfig
    taxonomy: TaxonomyConfig


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_app_config(root_dir: Path | None = None) -> AppConfig:
    base_dir = root_dir or Path(__file__).resolve().parents[2]
    config_dir = base_dir / "configs"
    load_dotenv(base_dir / ".env")

    llm = LLMSettings(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    pipeline = PipelineConfig(**_load_yaml(config_dir / "pipeline.yaml"))
    prompts = PromptConfig(**_load_yaml(config_dir / "prompts.yaml"))
    generation = GenerationConfig(**_load_yaml(config_dir / "generation.yaml"))
    sampling = SamplingConfig(**_load_yaml(config_dir / "sampling.yaml"))
    taxonomy = TaxonomyConfig(**_load_yaml(config_dir / "taxonomy.yaml"))

    return AppConfig(
        root_dir=base_dir,
        llm=llm,
        pipeline=pipeline,
        prompts=prompts,
        generation=generation,
        sampling=sampling,
        taxonomy=taxonomy,
    )
