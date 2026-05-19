from __future__ import annotations

import json

from ..llm.client import OpenAICompatibleClient
from ..enums import QueryType
from ..types import QueryGenerationResponse


class QueryGenerationAgent:
    def __init__(self, client: OpenAICompatibleClient, prompt: str) -> None:
        self._client = client
        self._prompt = prompt

    def _generate(self, payload: dict[str, object]) -> list[str]:
        user_prompt = json.dumps(
            payload,
            ensure_ascii=True,
        )
        response = self._client.generate_structured(
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=QueryGenerationResponse,
        )
        return response.queries

    def generate_from_seed(
        self,
        *,
        seed_topic: str,
        domain: str,
        target_query_type: QueryType,
        difficulty_hint: str | None,
        num_queries: int,
        type_definition: str,
        generation_tips: list[str],
    ) -> list[str]:
        return self._generate(
            {
                "seed_topic": seed_topic,
                "domain": domain,
                "target_query_type": target_query_type.value,
                "difficulty_hint": difficulty_hint,
                "num_queries": num_queries,
                "type_definition": type_definition,
                "generation_tips": generation_tips,
            }
        )

    def rewrite_query(
        self,
        *,
        source_query: str,
        domain: str,
        target_query_type: QueryType,
        difficulty_hint: str | None,
        num_queries: int,
        type_definition: str,
        generation_tips: list[str],
    ) -> list[str]:
        return self._generate(
            {
                "source_query": source_query,
                "domain": domain,
                "target_query_type": target_query_type.value,
                "difficulty_hint": difficulty_hint,
                "num_queries": num_queries,
                "type_definition": type_definition,
                "generation_tips": generation_tips,
            }
        )

    def generate_blindspot(
        self,
        *,
        blindspot: str,
        domain: str,
        target_query_type: QueryType,
        num_queries: int,
        type_definition: str,
        generation_tips: list[str],
    ) -> list[str]:
        return self._generate(
            {
                "blindspot": blindspot,
                "domain": domain,
                "target_query_type": target_query_type.value,
                "num_queries": num_queries,
                "type_definition": type_definition,
                "generation_tips": generation_tips,
            }
        )
