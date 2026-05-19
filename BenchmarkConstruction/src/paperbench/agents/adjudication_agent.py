from __future__ import annotations

import json

from ..llm.client import OpenAICompatibleClient
from ..types import AdjudicationResponse, QueryRecord


class AdjudicationAgent:
    def __init__(self, client: OpenAICompatibleClient, prompt: str) -> None:
        self._client = client
        self._prompt = prompt

    def adjudicate(self, record: QueryRecord) -> AdjudicationResponse:
        user_prompt = json.dumps(
            {
                "query": record.surface_form or record.raw_query,
                "canonical_query": record.canonical_query,
                "query_type": record.query_type.value if record.query_type else None,
                "topics": record.topics,
                "methods": record.methods,
                "applications": record.applications,
                "analysis_targets": record.analysis_targets,
                "scope": record.scope.value if record.scope else None,
                "difficulty": record.difficulty.value if record.difficulty else None,
                "domain": record.domain,
                "quality_checks": {
                    "is_decidable": record.is_decidable,
                    "has_clear_boundary": record.has_clear_boundary,
                    "is_non_degenerate": record.is_non_degenerate,
                    "no_fulltext_dependency": record.no_fulltext_dependency,
                },
            },
            ensure_ascii=True,
        )
        return self._client.generate_structured(
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=AdjudicationResponse,
        )
