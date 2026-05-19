from __future__ import annotations

import json

from ..llm.client import OpenAICompatibleClient
from ..types import DedupJudgeResponse, QueryRecord


class DedupJudgeAgent:
    def __init__(self, client: OpenAICompatibleClient, prompt: str) -> None:
        self._client = client
        self._prompt = prompt

    def judge(self, left: QueryRecord, right: QueryRecord) -> DedupJudgeResponse:
        user_prompt = json.dumps(
            {
                "query_a": left.surface_form or left.raw_query,
                "query_b": right.surface_form or right.raw_query,
                "canonical_a": left.canonical_query,
                "canonical_b": right.canonical_query,
            },
            ensure_ascii=True,
        )
        return self._client.generate_structured(
            system_prompt=self._prompt,
            user_prompt=user_prompt,
            response_model=DedupJudgeResponse,
        )
