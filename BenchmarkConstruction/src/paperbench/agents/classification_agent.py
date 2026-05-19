from __future__ import annotations

import json

from ..llm.client import OpenAICompatibleClient
from ..types import ClassificationResponse, QueryRecord


class ClassificationAgent:
    def __init__(self, client: OpenAICompatibleClient, prompt: str) -> None:
        self._client = client
        self._prompt = prompt

    def classify(self, record: QueryRecord) -> ClassificationResponse:
        user_prompt = json.dumps(
            {
                "query": record.surface_form or record.raw_query,
                "canonical_query": record.canonical_query,
            },
            ensure_ascii=True,
        )
        return self._client.generate_structured(
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=ClassificationResponse,
        )
