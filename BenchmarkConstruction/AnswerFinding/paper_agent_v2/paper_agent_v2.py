import asyncio
import json
import os
import re
import threading
from typing import Any, Optional

import httpx

DEFAULT_AGENT_MODEL_NAME = "Qwen3-4b-instruct"
DEFAULT_AGENT_API_KEY = "Qwen3-4b-instruct"
DEFAULT_AGENT_API_BASES = (
    "http://localhost:8998/v1",
    "http://localhost:8999/v1",
)
DEFAULT_WARMUP_MODEL_NAME = "qwen3-max"
DEFAULT_WARMUP_API_KEY = "sk-d85cf3f774b646659b6fd099b9c7672d"
DEFAULT_WARMUP_AGENT_API_BASES = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

try:
    from .prompts import PAPERSEARCH_SYSTEM_PROMPT, PAPERSEARCH_TOOL_SCHEMAS, PAPERSEARCH_USER_PROMPT, SELECT_PROMPT
    from .utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id
    from .http_retry import httpx_request_with_retry
except ImportError:
    from prompts import PAPERSEARCH_SYSTEM_PROMPT, PAPERSEARCH_TOOL_SCHEMAS, PAPERSEARCH_USER_PROMPT, SELECT_PROMPT
    from utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id
    from http_retry import httpx_request_with_retry

model_name = os.getenv("PAPER_AGENT_V2_MODEL_NAME", DEFAULT_AGENT_MODEL_NAME)
api_key = os.getenv("PAPER_AGENT_V2_API_KEY", DEFAULT_AGENT_API_KEY)
api_base = os.getenv("PAPER_AGENT_V2_API_BASE", DEFAULT_AGENT_API_BASES[0])

DEFAULT_SCORER_URLS = (
    "http://localhost:8993/classify",
    "http://localhost:8996/classify",
)


def _parse_backends(
    *,
    raw_urls: str,
    raw_names: str,
    legacy_url: str,
    default_urls: tuple[str, ...],
    default_prefix: str,
) -> list[dict[str, Any]]:
    urls = [item.strip() for item in re.split(r"[\s,]+", raw_urls) if item.strip()]
    if not urls:
        urls = [legacy_url] if legacy_url else list(default_urls)

    names = [item.strip() for item in re.split(r"[\s,]+", raw_names) if item.strip()]
    backends: list[dict[str, Any]] = []
    for index, url in enumerate(urls):
        backends.append(
            {
                "name": names[index] if index < len(names) and names[index] else f"{default_prefix}-{index + 1}",
                "url": url,
                "requests": 0,
            }
        )
    return backends


def _parse_agent_backends() -> list[dict[str, Any]]:
    return _parse_backends(
        raw_urls=os.getenv("PAPER_AGENT_V2_AGENT_URLS", "").strip(),
        raw_names=os.getenv("PAPER_AGENT_V2_AGENT_NAMES", "").strip(),
        legacy_url=os.getenv("PAPER_AGENT_V2_API_BASE", "").strip(),
        default_urls=DEFAULT_AGENT_API_BASES,
        default_prefix="agent",
    )


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except Exception:
        return default


def _clone_backends(backends: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item["name"]),
            "url": str(item["url"]),
            "requests": int(item.get("requests", 0)),
        }
        for item in backends
    ]


def _parse_scorer_backends() -> list[dict[str, Any]]:
    return _parse_backends(
        raw_urls=os.getenv("PAPER_AGENT_V2_SCORER_URLS", "").strip(),
        raw_names=os.getenv("PAPER_AGENT_V2_SCORER_NAMES", "").strip(),
        legacy_url=os.getenv("PAPER_AGENT_V2_SELECTOR_URL", "").strip(),
        default_urls=DEFAULT_SCORER_URLS,
        default_prefix="scorer",
    )

class PaperSearchV2Agent:
    def __init__(self, logger: Any = None, *args, **kwargs):
        self.logger = logger
        self.round_logger = kwargs.get("round_logger")
        self.session_id = str(kwargs.get("session_id", "") or "")
        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.history_search_queries: dict[str, int] = {}
        self.history_actions: list[tuple[str, str]] = []
        self.user_query = ""

        self.current_step = 0
        self.max_steps = kwargs.get("max_steps", 15)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        self.max_search_step = kwargs.get("max_search_step", 30)
        self.target_paper = kwargs.get("target_paper", 200)
        self.max_tolerance = kwargs.get("max_tolerance", 3)

        self.search_top_k = kwargs.get("search_top_k", 30)
        self.citations_limit = kwargs.get("citations_limit", 30)
        self.references_limit = kwargs.get("references_limit", -1)
        self.min_candidate_score = kwargs.get("min_candidate_score", 0.01)
        self.reward_top_k = kwargs.get("reward_top_k", 3)
        self.search_cost = kwargs.get("search_cost", 0.0)
        self.expand_cost = kwargs.get("expand_cost", 0.0)
        self.tool_schemas = PAPERSEARCH_TOOL_SCHEMAS

        self.client = PaperSearchV2Client(timeout=30.0)
        self.selector_client = httpx.AsyncClient(timeout=10.0)

        self.model_name = model_name
        self.api_key = api_key
        agent_backends = kwargs.get("agent_backends")
        self.agent_backends: list[dict[str, Any]] = [
            {
                "name": str(item["name"]),
                "url": str(item["url"]),
                "requests": int(item.get("requests", 0)),
            }
            for item in (agent_backends or _parse_agent_backends())
        ]
        self._agent_lock = threading.Lock()
        self._agent_tiebreak = 0
        self.api_base = self.agent_backends[0]["url"] if self.agent_backends else api_base
        self.warmup_enabled = _parse_bool_env("PAPER_AGENT_V2_WARMUP_ENABLED", False)
        self.warmup_top_k = max(0, _parse_int_env("PAPER_AGENT_V2_WARMUP_TOP_K", 0))
        self.warmup_model_name = (
            os.getenv("PAPER_AGENT_V2_WARMUP_MODEL_NAME", DEFAULT_WARMUP_MODEL_NAME).strip()
            or self.model_name
        )
        self.warmup_api_key = (
            os.getenv("PAPER_AGENT_V2_WARMUP_API_KEY", DEFAULT_WARMUP_API_KEY).strip()
            or self.api_key
        )
        warmup_agent_backends = kwargs.get("warmup_agent_backends")
        if warmup_agent_backends is not None:
            self.warmup_agent_backends = _clone_backends(warmup_agent_backends)
        elif (
            os.getenv("PAPER_AGENT_V2_WARMUP_AGENT_URLS", "").strip()
            or os.getenv("PAPER_AGENT_V2_WARMUP_AGENT_NAMES", "").strip()
            or os.getenv("PAPER_AGENT_V2_WARMUP_API_BASE", "").strip()
        ):
            self.warmup_agent_backends = _parse_backends(
                raw_urls=os.getenv("PAPER_AGENT_V2_WARMUP_AGENT_URLS", "").strip(),
                raw_names=os.getenv("PAPER_AGENT_V2_WARMUP_AGENT_NAMES", "").strip(),
                legacy_url=os.getenv("PAPER_AGENT_V2_WARMUP_API_BASE", "").strip(),
                default_urls=DEFAULT_WARMUP_AGENT_API_BASES,
                default_prefix="warmup-agent",
            )
        else:
            self.warmup_agent_backends = _clone_backends(self.agent_backends)
        self._warmup_agent_lock = threading.Lock()
        self._warmup_agent_tiebreak = 0

        scorer_backends = kwargs.get("scorer_backends")
        self.scorer_backends: list[dict[str, Any]] = [
            {
                "name": str(item["name"]),
                "url": str(item["url"]),
                "requests": int(item.get("requests", 0)),
            }
            for item in (scorer_backends or _parse_scorer_backends())
        ]
        self._scorer_lock = threading.Lock()
        self._scorer_tiebreak = 0

        self.selector_url = self.scorer_backends[0]["url"] if self.scorer_backends else ""
        self.selector_model_name = kwargs.get(
            "selector_model_name", os.getenv("PAPER_AGENT_V2_SELECTOR_MODEL", "selector")
        )

        self.status = "idle"
        self.is_paused = False
        self.stop_requested = False
        self.executed_round_count = 0
        self.warmup_rounds_completed = 0

    async def close(self) -> None:
        await self.selector_client.aclose()
        await self.client.close()

    def _log_info(self, message: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(message, *args)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, default=str, ensure_ascii=True))
        except Exception:
            return str(value)

    def _tool_calls_to_records(self, tool_calls: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tool_call in tool_calls or []:
            function = getattr(tool_call, "function", None)
            records.append(
                {
                    "id": getattr(tool_call, "id", ""),
                    "type": getattr(tool_call, "type", ""),
                    "name": getattr(function, "name", ""),
                    "arguments": getattr(function, "arguments", ""),
                }
            )
        return records

    def _round_state_summary(self) -> dict[str, Any]:
        with self._paper_pool_lock:
            pool_size = len(self.paper_pool.papers)
        return {
            "current_step": self.current_step,
            "executed_round_count": self.executed_round_count,
            "warmup_rounds_completed": self.warmup_rounds_completed,
            "warmup_enabled": self.warmup_enabled,
            "warmup_top_k": self.warmup_top_k,
            "max_steps": self.max_steps,
            "status": self.status,
            "search_count": sum(1 for action, _ in self.history_actions if action == "search"),
            "expand_count": sum(1 for action, _ in self.history_actions if action == "expand"),
            "paper_pool_size": pool_size,
        }

    def _log_round_event(self, event_type: str, **payload: Any) -> None:
        if not self.round_logger:
            return
        record = {
            "session_id": self.session_id,
            "user_query": self.user_query,
            **payload,
        }
        self.round_logger.log_event(event_type=event_type, payload=self._json_safe(record))

    @staticmethod
    def _normalize_arxiv_id(arxiv_id: str) -> str:
        return normalize_arxiv_id(arxiv_id)

    @staticmethod
    def _copy_paper_with_arxiv_id(paper: Paper, arxiv_id: str) -> Paper:
        update = {"arxiv_id": arxiv_id, "paper_id": arxiv_id or paper.paper_id}
        try:
            return paper.model_copy(update=update)
        except AttributeError:
            return paper.copy(update=update)

    @staticmethod
    def _normalize_tool_argument(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    def _reserve_backend(
        self,
        backends: list[dict[str, Any]],
        lock: threading.Lock,
        tiebreak_attr: str,
        *,
        excluded: set[str] | None = None,
        backend_kind: str,
    ) -> dict[str, Any]:
        excluded = excluded or set()
        with lock:
            available = [
                (index, backend)
                for index, backend in enumerate(backends)
                if backend["url"] not in excluded
            ]
            if not available:
                raise RuntimeError(f"No {backend_kind} backends available")

            min_requests = min(int(backend["requests"]) for _, backend in available)
            candidate_indexes = [
                index for index, backend in available if int(backend["requests"]) == min_requests
            ]
            start = getattr(self, tiebreak_attr) % len(backends)

            selected_index = candidate_indexes[0]
            for offset in range(len(backends)):
                probe_index = (start + offset) % len(backends)
                if probe_index in candidate_indexes:
                    selected_index = probe_index
                    break

            backend = backends[selected_index]
            backend["requests"] = int(backend["requests"]) + 1
            setattr(self, tiebreak_attr, (selected_index + 1) % len(backends))
            return {"name": backend["name"], "url": backend["url"], "requests": backend["requests"]}

    def _reserve_agent(self, excluded: set[str] | None = None) -> dict[str, Any]:
        return self._reserve_backend(
            self.agent_backends,
            self._agent_lock,
            "_agent_tiebreak",
            excluded=excluded,
            backend_kind="agent",
        )

    def _reserve_warmup_agent(self, excluded: set[str] | None = None) -> dict[str, Any]:
        return self._reserve_backend(
            self.warmup_agent_backends,
            self._warmup_agent_lock,
            "_warmup_agent_tiebreak",
            excluded=excluded,
            backend_kind="warmup_agent",
        )

    def _reserve_scorer(self, excluded: set[str] | None = None) -> dict[str, Any]:
        return self._reserve_backend(
            self.scorer_backends,
            self._scorer_lock,
            "_scorer_tiebreak",
            excluded=excluded,
            backend_kind="scorer",
        )

    def _format_history_actions(self) -> str:
        if not self.history_actions:
            return "None"

        lines: list[str] = []
        for action, value in self.history_actions:
            if action == "search":
                lines.append(f"[Search] {value}")
            elif action == "expand":
                lines.append(f"[Expand] {value}")
            else:
                raise ValueError(f"Invalid action: {action}")
        return "\n".join(lines)

    def _clean_user_prompt(self, user_prompt: str) -> str:
        return re.sub(r"[\u200b\u200c\u200d\uFEFF\u00A0]", " ", user_prompt)

    def _is_warmup_round(self, round_index: int) -> bool:
        return self.warmup_enabled and round_index <= self.warmup_top_k

    def _get_round_phase(self, round_index: int) -> str:
        return "warmup" if self._is_warmup_round(round_index) else "base"

    def _get_round_model_config(self, round_index: int) -> dict[str, Any]:
        if self._is_warmup_round(round_index):
            return {
                "model_phase": "warmup",
                "model_name": self.warmup_model_name,
                "api_key": self.warmup_api_key,
                "agent_backends": self.warmup_agent_backends,
            }
        return {
            "model_phase": "base",
            "model_name": self.model_name,
            "api_key": self.api_key,
            "agent_backends": self.agent_backends,
        }

    def _reserve_agent_for_phase(self, model_phase: str, excluded: set[str] | None = None) -> dict[str, Any]:
        if model_phase == "warmup":
            return self._reserve_warmup_agent(excluded=excluded)
        return self._reserve_agent(excluded=excluded)

    def _get_next_turn_message(self, user_query: str, round_index: int = 1):
        user_prompt = PAPERSEARCH_USER_PROMPT.format(
            user_query=user_query,
            paper_list=self.paper_pool.paper_list,
            history_actions=self._format_history_actions(),
        )
        user_prompt = self._clean_user_prompt(user_prompt)

        last_exc: Optional[BaseException] = None
        attempted_urls: set[str] = set()
        model_config = self._get_round_model_config(round_index)
        model_phase = str(model_config["model_phase"])
        model_name = str(model_config["model_name"])
        api_key = str(model_config["api_key"])
        agent_backends = list(model_config["agent_backends"])

        for _ in range(len(agent_backends)):
            backend = self._reserve_agent_for_phase(model_phase, excluded=attempted_urls)
            attempted_urls.add(backend["url"])
            self._log_round_event(
                "round_request",
                round_index=round_index,
                model_phase=model_phase,
                effective_model_name=model_name,
                backend=backend,
                prompt=user_prompt,
            )
            try:
                response = call_openai_chat(
                    model_name=model_name,
                    system_prompt=PAPERSEARCH_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    api_key=api_key,
                    api_base=backend["url"],
                    tools=self.tool_schemas,
                    tool_choice="required",
                )
                msg = response.choices[0].message
                tool_call_records = self._tool_calls_to_records(getattr(msg, "tool_calls", None))
                self._log_round_event(
                    "round_response",
                    round_index=round_index,
                    model_phase=model_phase,
                    effective_model_name=model_name,
                    backend=backend,
                    response_content=self._json_safe(getattr(msg, "content", "")),
                    tool_calls=tool_call_records,
                )
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return msg, msg.tool_calls
                return msg, None
            except Exception as exc:
                last_exc = exc
                self._log_info(
                    "Search agent %s failed after %s requests: %r",
                    backend["name"],
                    backend["requests"],
                    exc,
                )
                self._log_round_event(
                    "round_error",
                    round_index=round_index,
                    model_phase=model_phase,
                    effective_model_name=model_name,
                    backend=backend,
                    error=str(exc),
                )

        self._log_info("Failed to get next turn message: %s", last_exc)
        return None, None

    async def _execute_tool_calls(self, tool_calls: list, round_index: int, **kwargs) -> tuple[float, list[dict[str, Any]]]:
        tasks = []
        summaries: list[dict[str, Any]] = []
        seen_search_queries: set[str] = set()
        seen_expand_ids: set[str] = set()

        for tool_call in tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except Exception as exc:
                self._log_info("Failed to parse tool arguments: %s", exc)
                continue

            if tool_call.function.name == "search":
                query = self._normalize_tool_argument(tool_args.get("query"))
                if query and query not in seen_search_queries:
                    seen_search_queries.add(query)
                    tasks.append(self.search(query, **kwargs))
                    summaries.append({"name": "search", "arguments": {"query": query}})
                    self.history_actions.append(("search", query))
            elif tool_call.function.name == "expand":
                arxiv_id = self._normalize_tool_argument(tool_args.get("arxiv_id"))
                arxiv_id = self._normalize_arxiv_id(arxiv_id)
                if arxiv_id and arxiv_id not in seen_expand_ids:
                    seen_expand_ids.add(arxiv_id)
                    tasks.append(self.expand(arxiv_id, **kwargs))
                    summaries.append({"name": "expand", "arguments": {"arxiv_id": arxiv_id}})
                    self.history_actions.append(("expand", arxiv_id))

        if not tasks:
            self._log_round_event("tool_execution", round_index=round_index, tools=summaries)
            return 0.0, summaries

        scores = await asyncio.gather(*tasks)
        for summary, score in zip(summaries, scores):
            summary["reward"] = float(score)
        self._log_round_event("tool_execution", round_index=round_index, tools=summaries)
        return float(sum(scores)), summaries

    async def run(self, user_query: str, **kwargs):
        self.user_query = user_query
        self.status = "ITERATING"
        self.stop_requested = False
        self.is_paused = False

        self.current_step = 0
        self.executed_round_count = 0
        self.warmup_rounds_completed = 0
        tolerance = 0
        last_paper_count = len(self.paper_pool)
        remaining_search_steps = self.max_search_step
        break_flag = False
        self._log_round_event(
            "run_started",
            round_index=0,
            max_steps=self.max_steps,
            warmup_enabled=self.warmup_enabled,
            warmup_top_k=self.warmup_top_k,
            max_parallel_calls=self.max_parallel_calls,
            state=self._round_state_summary(),
        )

        while self.current_step < self.max_steps or (
            self.warmup_enabled and self.warmup_rounds_completed < self.warmup_top_k
        ):
            if self.stop_requested:
                break

            while self.is_paused:
                self.status = "PAUSED"
                await asyncio.sleep(1)
                if self.stop_requested:
                    break

            if self.stop_requested:
                break

            if self.status == "PAUSED":
                self.status = "ITERATING"

            round_index = self.executed_round_count + 1
            model_phase = self._get_round_phase(round_index)
            msg, tool_calls = self._get_next_turn_message(user_query, round_index)
            if msg is None and tool_calls is None:
                self._log_round_event(
                    "round_summary",
                    round_index=round_index,
                    model_phase=model_phase,
                    break_flag=True,
                    break_reason="no_model_response",
                    tolerance=tolerance,
                    remaining_search_steps=remaining_search_steps,
                    state=self._round_state_summary(),
                )
                break

            if not tool_calls:
                tolerance += 1
                if tolerance >= self.max_tolerance:
                    break_flag = True
            else:
                tool_calls = tool_calls[: self.max_parallel_calls]
                _, tool_call_summaries = await self._execute_tool_calls(tool_calls, round_index=round_index, **kwargs)

                for tool_call in tool_call_summaries:
                    if tool_call["name"] == "search":
                        remaining_search_steps -= 1
                        if remaining_search_steps <= 0:
                            break_flag = True
                            break

                new_paper_count = len(self.paper_pool)
                if new_paper_count > last_paper_count:
                    tolerance = 0
                else:
                    tolerance += 1
                last_paper_count = new_paper_count

                if new_paper_count >= self.target_paper or tolerance >= self.max_tolerance:
                    break_flag = True

            self.executed_round_count += 1
            if model_phase == "warmup":
                self.warmup_rounds_completed += 1
            else:
                self.current_step += 1
            self._log_round_event(
                "round_summary",
                round_index=round_index,
                model_phase=model_phase,
                break_flag=break_flag,
                tolerance=tolerance,
                remaining_search_steps=remaining_search_steps,
                tool_call_count=len(tool_calls or []),
                state=self._round_state_summary(),
            )
            if break_flag:
                break

        self.status = "COMPLETED"
        self._log_round_event(
            "run_completed",
            round_index=self.executed_round_count,
            state=self._round_state_summary(),
        )
        return self.paper_pool.get_order_papers()

    def pause(self):
        self.is_paused = True
        self.status = "PAUSED"

    def resume(self):
        self.is_paused = False
        self.status = "ITERATING"

    def stop(self):
        self.stop_requested = True
        self.status = "COMPLETED"

    def get_state(self):
        ranked_papers = self.paper_pool.get_ranked_papers()
        papers_data = []
        for entry in ranked_papers:
            if entry.score <= 0.5:
                continue

            paper = entry.paper
            papers_data.append(
                {
                    "id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": str(paper.year) if paper.year else "N/A",
                    "abstract": paper.abstract,
                    "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
                    "isExpanded": entry.expand,
                    "isExpanding": entry.expanding,
                    "score": entry.score,
                    "source": entry.source,
                    "origin": entry.origin,
                }
            )

        return {
            "status": self.status,
            "agentState": {
                "currentStep": self.current_step,
                "totalSteps": self.max_steps,
                "warmupEnabled": self.warmup_enabled,
                "warmupTopK": self.warmup_top_k,
                "warmupCompleted": self.warmup_rounds_completed,
                "totalRounds": self.executed_round_count,
                "searchCount": sum(1 for action, _ in self.history_actions if action == "search"),
                "expandCount": sum(1 for action, _ in self.history_actions if action == "expand"),
                "paperCount": len(papers_data),
                "scannedCount": len(ranked_papers),
                "isSearching": self.status == "ITERATING",
            },
            "papers": papers_data,
        }

    def _paper_is_usable(self, paper: Optional[Paper]) -> bool:
        if paper is None:
            return False
        return bool(self._normalize_arxiv_id(paper.arxiv_id))

    def _dedupe_and_normalize_candidates(self, papers: list[Paper]) -> list[Paper]:
        normalized_papers: list[Paper] = []
        seen_base_ids: set[str] = set()

        for paper in papers:
            base_id = self._normalize_arxiv_id(paper.arxiv_id)
            if not base_id or base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)

            with self._paper_pool_lock:
                if self.paper_pool.has_paper(base_id):
                    continue

            normalized_papers.append(self._copy_paper_with_arxiv_id(paper, base_id))

        return normalized_papers

    async def search(self, query: str, **kwargs) -> float:
        if query in self.history_search_queries:
            return -0.5

        self._log_info("[Search]: %s", query)
        try:
            papers = await self.client.search(query=query, limit=self.search_top_k, source="local_db")
        except Exception as exc:
            self._log_info("Error in local_db search %s: %r", query, exc)
            self.history_search_queries[query] = 0
            return 0.0

        candidate_papers = self._dedupe_and_normalize_candidates([paper for paper in papers if self._paper_is_usable(paper)])
        relevance_scores = await asyncio.gather(
            *(self.get_relevance_score(self.user_query, paper, **kwargs) for paper in candidate_papers)
        ) if candidate_papers else []

        kept_scores: list[float] = []
        kept_count = 0
        for paper, score in zip(candidate_papers, relevance_scores):
            if score < self.min_candidate_score:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.arxiv_id):
                    continue
                self.paper_pool.add_paper(paper, "search", query, score)
            kept_scores.append(score)
            kept_count += 1

        self.history_search_queries[query] = kept_count
        if not kept_scores:
            return -0.2

        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.search_cost

    async def expand(self, arxiv_id: str, **kwargs) -> float:
        base_id = self._normalize_arxiv_id(arxiv_id)
        with self._paper_pool_lock:
            paper_pool_entry = self.paper_pool.get_paper(base_id)
            if not paper_pool_entry or paper_pool_entry.expand:
                return -0.5
            paper_pool_entry.expand = True
            paper_pool_entry.expanding = True

        root_paper_id = (
            paper_pool_entry.paper.raw_paper_id or paper_pool_entry.paper.paper_id or paper_pool_entry.paper.arxiv_id
        )

        try:
            citations, references = await asyncio.gather(
                self.client.get_citations(root_paper_id, limit=self.citations_limit),
                self.client.get_references(root_paper_id, limit=self.references_limit),
            )
        except Exception as exc:
            self._log_info("Error in expand %s: %r", base_id, exc)
            with self._paper_pool_lock:
                paper_pool_entry.expanding = False
            return 0.0

        merged_candidates: list[Paper] = []
        seen_candidate_ids: set[str] = set()
        for candidate in citations + references:
            candidate_key = candidate.raw_paper_id or candidate.paper_id or candidate.arxiv_id
            if not candidate_key or candidate_key == root_paper_id or candidate_key in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_key)
            merged_candidates.append(candidate)

        async def _hydrate_candidate(candidate: Paper) -> Optional[Paper]:
            target_id = candidate.raw_paper_id or candidate.paper_id or candidate.arxiv_id
            paper = candidate
            if not paper.abstract and target_id:
                try:
                    detail = await self.client.get_paper(target_id)
                except Exception as exc:
                    self._log_info("Error fetching detail for %s: %r", target_id, exc)
                    detail = None
                if detail is not None:
                    paper = detail

            normalized_id = self._normalize_arxiv_id(paper.arxiv_id)
            if not normalized_id or not paper.abstract:
                return None
            return self._copy_paper_with_arxiv_id(paper, normalized_id)

        hydrated_candidates = await asyncio.gather(*(_hydrate_candidate(candidate) for candidate in merged_candidates))
        candidate_papers = self._dedupe_and_normalize_candidates([paper for paper in hydrated_candidates if paper is not None])

        relevance_scores = await asyncio.gather(
            *(self.get_relevance_score(self.user_query, paper, **kwargs) for paper in candidate_papers)
        ) if candidate_papers else []

        kept_scores: list[float] = []
        for paper, score in zip(candidate_papers, relevance_scores):
            if score < self.min_candidate_score:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.arxiv_id):
                    continue
                origin = f"[{paper_pool_entry.paper.arxiv_id}] {paper_pool_entry.paper.title}"
                self.paper_pool.add_paper(paper, "expand", origin, score)
            kept_scores.append(score)

        with self._paper_pool_lock:
            paper_pool_entry.expanding = False

        if not kept_scores:
            return -0.2
        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.expand_cost

    async def get_relevance_score(self, query: str, paper: Paper, **kwargs) -> float:
        prompt = SELECT_PROMPT.format(title=paper.title, abstract=paper.abstract, user_query=query)
        payload = {
            "model": self.selector_model_name,
            "input": [prompt],
        }
        last_exc: Optional[BaseException] = None
        attempted_urls: set[str] = set()

        for _ in range(len(self.scorer_backends)):
            backend = self._reserve_scorer(excluded=attempted_urls)
            attempted_urls.add(backend["url"])
            try:
                response = await httpx_request_with_retry(
                    self.selector_client,
                    "POST",
                    backend["url"],
                    json=payload,
                    max_retries=3,
                )
                response.raise_for_status()
                body = response.json()
                predictions = body["data"]
                scores = [pred["probs"][0] for pred in predictions]
                return float(scores[0])
            except Exception as exc:
                last_exc = exc
                self._log_info(
                    "Scorer %s failed after %s requests: %r",
                    backend["name"],
                    backend["requests"],
                    exc,
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No scorer backend configured")
