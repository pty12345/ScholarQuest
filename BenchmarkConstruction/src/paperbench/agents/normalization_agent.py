from __future__ import annotations

import json

from ..llm.client import OpenAICompatibleClient
from ..types import NormalizationResponse


class NormalizationAgent:
    def __init__(self, client: OpenAICompatibleClient, prompt: str) -> None:
        self._client = client
        self._prompt = prompt

    def normalize(self, raw_query: str) -> NormalizationResponse:
        user_prompt = json.dumps({"raw_query": raw_query}, ensure_ascii=True)
        return self._client.generate_structured(
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=NormalizationResponse,
        )
