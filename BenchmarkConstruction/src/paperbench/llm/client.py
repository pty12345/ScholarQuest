from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

from ..io_utils import append_jsonl

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        cache_path: Path | None = None,
        temperature: float = 0.2,
        use_responses_api: bool = True,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._cache_path = cache_path
        self._use_responses_api = use_responses_api
        self._cache = self._load_cache(cache_path)
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def _load_cache(self, cache_path: Path | None) -> dict[str, str]:
        if cache_path is None or not cache_path.exists():
            return {}
        cache: dict[str, str] = {}
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    key = payload.get("cache_key")
                    value = payload.get("response_text")
                    if isinstance(key, str) and isinstance(value, str):
                        cache[key] = value
        return cache

    def _make_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        joined = f"{self._model}\n{system_prompt}\n{user_prompt}"
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _extract_text_from_responses_api(self, completion: Any) -> str:
        output_text = getattr(completion, "output_text", "")
        if isinstance(output_text, str):
            return output_text.strip()
        return ""

    def _generate_via_responses_api(self, system_prompt: str, user_prompt: str) -> str:
        completion = self._client.responses.create(
            model=self._model,
            temperature=self._temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = self._extract_text_from_responses_api(completion)
        if text:
            return text

        error = getattr(completion, "error", None)
        if error is not None:
            error_message = getattr(error, "message", None)
            if isinstance(error_message, str) and error_message.strip():
                raise ValueError(error_message.strip())
        raise ValueError("Responses API returned empty output text.")

    def _generate_via_chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        raise ValueError("Chat completions API returned empty content.")

    def _validate_json_text(self, response_model: type[T], text: str) -> T:
        validator = getattr(response_model, "model_validate_json", None)
        if callable(validator):
            return cast(T, validator(text))
        parser = getattr(response_model, "parse_raw", None)
        if callable(parser):
            return cast(T, parser(text))
        raise TypeError("Response model does not support JSON validation.")

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        cache_key = self._make_cache_key(system_prompt, user_prompt)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._validate_json_text(response_model, cached)

        if self._use_responses_api:
            try:
                text = self._generate_via_responses_api(system_prompt, user_prompt)
            except Exception:
                text = self._generate_via_chat_completions(system_prompt, user_prompt)
        else:
            text = self._generate_via_chat_completions(system_prompt, user_prompt)
        parsed = self._validate_json_text(response_model, text)

        if self._cache_path is not None:
            self._cache[cache_key] = text
            append_jsonl(
                self._cache_path,
                {
                    "cache_key": cache_key,
                    "model": self._model,
                    "response_text": text,
                },
            )
        return parsed
