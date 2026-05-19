from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from dotenv import load_dotenv

try:
    from .answer_finding_prompts import (
        FIRST_ROUND_REWRITE_SYSTEM_PROMPT,
        FIRST_ROUND_REWRITE_USER_PROMPT,
        SECOND_ROUND_REWRITE_SYSTEM_PROMPT,
        SECOND_ROUND_REWRITE_USER_PROMPT,
    )
    from .answer_finding_types import (
        AnswerFindingResult,
        InteractionLogResult,
        LoggedPaperResult,
        PaperJudgment,
        RankedPaperResult,
        RewriteQueriesResponse,
    )
    from .prompts import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT
    from .prompts import build_user_prompt as build_judge_user_prompt
    from .paper_agent_v2.backend_pool import BackendPool, parse_backends
    from .paper_agent_v2.http_retry import httpx_request_with_retry
    from .paper_agent_v2.utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id
except ImportError:
    from answer_finding_prompts import (
        FIRST_ROUND_REWRITE_SYSTEM_PROMPT,
        FIRST_ROUND_REWRITE_USER_PROMPT,
        SECOND_ROUND_REWRITE_SYSTEM_PROMPT,
        SECOND_ROUND_REWRITE_USER_PROMPT,
    )
    from answer_finding_types import (  # type: ignore[no-redef]
        AnswerFindingResult,
        InteractionLogResult,
        LoggedPaperResult,
        PaperJudgment,
        RankedPaperResult,
        RewriteQueriesResponse,
    )
    from prompts import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT  # type: ignore[no-redef]
    from prompts import build_user_prompt as build_judge_user_prompt  # type: ignore[no-redef]
    from paper_agent_v2.backend_pool import BackendPool, parse_backends
    from paper_agent_v2.http_retry import httpx_request_with_retry
    from paper_agent_v2.utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id


DEFAULT_REWRITE_MODEL = "qwen-max"
DEFAULT_REWRITE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_JUDGE_MODEL = "Qwen3-30B-A3B-Instruct-2507"
DEFAULT_JUDGE_API_BASES = (
    DEFAULT_REWRITE_API_BASE,
)
DEFAULT_SCORER_URLS = ("http://localhost:9000/classify",)
DEFAULT_SCORER_MODEL = "selector_qwen3_8b"
DEFAULT_SCORER_PREFILTER_THRESHOLD = 0.1
MIN_EXPAND_RELEVANCE_LEVEL = 1
MIN_ANSWER_RELEVANCE_LEVEL = 2
EXTRA_WEAK_CONFIDENCES = {"low", "medium"}
ANSWER_FINDING_ROOT = Path(__file__).resolve().parent
load_dotenv(ANSWER_FINDING_ROOT / ".env")


class AnswerFindingPipeline:
    def __init__(
        self,
        *,
        queries_per_round: int = 10,
        search_top_k: int = 10,
        selector_threshold: float = float(MIN_EXPAND_RELEVANCE_LEVEL - 1),
        expand_threshold: float = float(MIN_EXPAND_RELEVANCE_LEVEL - 1),
        citations_limit: int = 30,
        references_limit: int = -1,
        context_paper_count: int = 20,
        answer_score_threshold: float = float(MIN_ANSWER_RELEVANCE_LEVEL),
        max_answer_count: Optional[int] = 500,
        max_scored_paper_count: Optional[int] = 10000,
        scorer_prefilter_threshold: float = DEFAULT_SCORER_PREFILTER_THRESHOLD,
    ) -> None:
        self.queries_per_round = queries_per_round
        self.search_top_k = search_top_k
        self.selector_threshold = selector_threshold
        self.expand_threshold = expand_threshold
        self.citations_limit = citations_limit
        self.references_limit = references_limit
        self.context_paper_count = context_paper_count
        self.answer_score_threshold = answer_score_threshold
        self.max_answer_count = max_answer_count
        self.max_scored_paper_count = max_scored_paper_count
        self.scorer_prefilter_threshold = scorer_prefilter_threshold

        self.paper_pool = PaperPool(max_size=max(20, context_paper_count))
        self.paper_client = PaperSearchV2Client(timeout=30.0)
        self.scorer_client = httpx.AsyncClient(timeout=10.0)

        self.rewrite_model = os.getenv("ANSWER_FINDING_REWRITE_MODEL", DEFAULT_REWRITE_MODEL).strip()
        self.rewrite_api_base = os.getenv("ANSWER_FINDING_REWRITE_API_BASE", DEFAULT_REWRITE_API_BASE).strip()
        self.rewrite_api_key = (
            os.getenv("ANSWER_FINDING_REWRITE_API_KEY", "").strip()
            or os.getenv("PAPER_AGENT_V2_WARMUP_API_KEY", "").strip()
        )
        if not self.rewrite_api_key:
            raise ValueError("Missing rewrite API key. Set ANSWER_FINDING_REWRITE_API_KEY or PAPER_AGENT_V2_WARMUP_API_KEY.")

        self.judge_model = os.getenv("ANSWER_FINDING_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip()
        self.judge_backends = parse_backends(
            raw_urls=os.getenv("ANSWER_FINDING_JUDGE_API_BASES", "").strip(),
            raw_names=os.getenv("ANSWER_FINDING_JUDGE_API_NAMES", "").strip(),
            legacy_url=os.getenv("ANSWER_FINDING_JUDGE_API_BASE", "").strip(),
            default_urls=DEFAULT_JUDGE_API_BASES,
            default_prefix="judge",
        )
        self.judge_pool = BackendPool(self.judge_backends, backend_kind="judge")
        self.judge_api_key = (
            os.getenv("ANSWER_FINDING_JUDGE_API_KEY", "").strip()
            or self.rewrite_api_key
        )
        raw_scorer_urls = (
            os.getenv("ANSWER_FINDING_SCORER_URLS", "").strip()
            or os.getenv("PAPER_AGENT_V2_SCORER_URLS", "").strip()
        )
        raw_scorer_names = (
            os.getenv("ANSWER_FINDING_SCORER_NAMES", "").strip()
            or os.getenv("PAPER_AGENT_V2_SCORER_NAMES", "").strip()
        )
        self.scorer_backends = parse_backends(
            raw_urls=raw_scorer_urls,
            raw_names=raw_scorer_names,
            legacy_url=os.getenv("ANSWER_FINDING_SCORER_URL", "").strip(),
            default_urls=DEFAULT_SCORER_URLS,
            default_prefix="scorer",
        )
        self.scorer_pool = BackendPool(self.scorer_backends, backend_kind="scorer")
        self.selector_model_name = (
            os.getenv("ANSWER_FINDING_SELECTOR_MODEL", "").strip()
            or os.getenv("PAPER_AGENT_V2_SELECTOR_MODEL", DEFAULT_SCORER_MODEL).strip()
        )
        self.interaction_logs: list[InteractionLogResult] = []
        self.paper_judgment_cache: dict[str, PaperJudgment] = {}
        self.prefilter_rejected_ids: set[str] = set()
        self.prefilter_passed_ids: set[str] = set()
        self.duplicate_scored_paper_ids: set[str] = set()

    async def close(self) -> None:
        await self.scorer_client.aclose()
        await self.paper_client.close()

    async def run(self, query: str) -> AnswerFindingResult:
        self.interaction_logs = []
        self.paper_judgment_cache = {}
        self.prefilter_rejected_ids = set()
        self.prefilter_passed_ids = set()
        self.duplicate_scored_paper_ids = set()
        first_round_queries = self.rewrite_queries(query)
        first_round_seed_ids = await self.search_and_score_queries(query, first_round_queries)
        await self.expand_until_converged(query, first_round_seed_ids)

        return AnswerFindingResult(
            query=query,
            first_round_queries=first_round_queries,
            second_round_queries=[],
            scored_paper_count=len(self.paper_judgment_cache),
            scorer_checked_count=len(self.prefilter_rejected_ids) + len(self.prefilter_passed_ids),
            prefilter_rejected_count=len(self.prefilter_rejected_ids),
            prefilter_passed_count=len(self.prefilter_passed_ids),
            duplicate_scored_paper_count=len(self.duplicate_scored_paper_ids),
            weakly_relevant_count=self.weakly_relevant_count(),
            weak_low_medium_count=self.weak_low_medium_count(),
            logs=list(self.interaction_logs),
            papers=self.export_ranked_results(),
        )

    def rewrite_queries(
        self,
        query: str,
        *,
        paper_context: Optional[str] = None,
        previous_queries: Optional[list[str]] = None,
    ) -> list[str]:
        if paper_context is None:
            system_prompt = FIRST_ROUND_REWRITE_SYSTEM_PROMPT
            user_prompt = FIRST_ROUND_REWRITE_USER_PROMPT.format(query=query)
            excluded: set[str] = set()
        else:
            system_prompt = SECOND_ROUND_REWRITE_SYSTEM_PROMPT
            user_prompt = SECOND_ROUND_REWRITE_USER_PROMPT.format(
                query=query,
                previous_queries="\n".join(f"- {item}" for item in (previous_queries or [])),
                paper_context=paper_context,
            )
            excluded = {item.strip().lower() for item in (previous_queries or []) if item.strip()}

        response = call_openai_chat(
            model_name=self.rewrite_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=self.rewrite_api_key,
            api_base=self.rewrite_api_base,
        )
        message_content = response.choices[0].message.content or ""
        parsed = self._parse_rewrite_response(message_content)
        return self._normalize_queries(parsed.queries, fallback_query=query, excluded=excluded)

    def _parse_rewrite_response(self, raw_content: str) -> RewriteQueriesResponse:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
        payload = json.loads(content)
        queries = payload.get("queries", []) if isinstance(payload, dict) else []
        if not isinstance(queries, list):
            raise ValueError("Rewrite response must contain a queries list.")
        return RewriteQueriesResponse(queries=[str(item) for item in queries])

    def _normalize_queries(
        self,
        queries: Iterable[str],
        *,
        fallback_query: str,
        excluded: set[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set(excluded)

        for query in queries:
            cleaned = " ".join(str(query).split()).strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            normalized.append(cleaned)
            seen.add(lowered)

        fallback_cleaned = " ".join(fallback_query.split()).strip()
        if fallback_cleaned and fallback_cleaned.lower() not in seen:
            normalized.append(fallback_cleaned)
            seen.add(fallback_cleaned.lower())

        while normalized and len(normalized) < self.queries_per_round:
            normalized.append(normalized[-1])

        if not normalized:
            normalized = [fallback_cleaned] * self.queries_per_round

        return normalized[: self.queries_per_round]

    async def search_and_score_queries(self, user_query: str, queries: list[str]) -> list[str]:
        seed_ids: list[str] = []
        for query in queries:
            if self._should_stop():
                break
            source = f"search:{query}"
            try:
                papers = await self.paper_client.search(query=query, limit=self.search_top_k, source="local_db")
            except Exception:
                self._append_interaction_log(source, [])
                continue

            loggable_papers = self._normalize_loggable_papers(papers)
            if not loggable_papers:
                self._append_interaction_log(source, [])
                continue

            scored_papers, judgments = await self.score_loggable_papers(user_query, loggable_papers)
            self._append_interaction_log(source, self._build_logged_results(scored_papers, judgments))

            judgment_by_arxiv_id = {paper.arxiv_id: judgment for paper, judgment in zip(scored_papers, judgments)}
            candidate_papers = self._dedupe_and_normalize_candidates(scored_papers)
            for paper in candidate_papers:
                score = float(judgment_by_arxiv_id[paper.arxiv_id].relevance_level)
                if score <= self.selector_threshold or self.paper_pool.has_paper(paper.arxiv_id):
                    continue
                self.paper_pool.add_paper(paper, "search", query, score)
                seed_ids.append(paper.arxiv_id)
                if self._should_stop():
                    break
        return list(dict.fromkeys(seed_ids))

    async def expand_two_hops(self, user_query: str, seed_ids: list[str], *, round_label: str) -> None:
        first_hop_ids: list[str] = []
        round_candidates: list[LoggedPaperResult] = []
        for seed_id in seed_ids:
            new_ids, logged_papers = await self.expand_once(user_query, seed_id)
            first_hop_ids.extend(new_ids)
            round_candidates.extend(logged_papers)

        # second_hop_seeds = list(dict.fromkeys(first_hop_ids))
        # for seed_id in second_hop_seeds:
        #     _, logged_papers = await self.expand_once(user_query, seed_id)
        #     round_candidates.extend(logged_papers)

        self._append_interaction_log(f"expand:{round_label}", self._dedupe_logged_results(round_candidates))

    async def expand_until_converged(self, user_query: str, _initial_seed_ids: list[str]) -> None:
        iteration = 0
        while not self._should_stop():
            seed_ids = self._pending_expand_ids(limit=1)
            if not seed_ids:
                return
            source_label = "expand:initial" if iteration == 0 else f"expand:iter{iteration}"
            await self.expand_papers(user_query, seed_ids, source_label=source_label)
            iteration += 1

    async def expand_papers(self, user_query: str, seed_ids: list[str], *, source_label: str) -> None:
        round_candidates: list[LoggedPaperResult] = []
        for seed_id in dict.fromkeys(seed_ids):
            if self._should_stop():
                break
            _, logged_papers = await self.expand_once(user_query, seed_id)
            round_candidates.extend(logged_papers)
        self._append_interaction_log(source_label, self._dedupe_logged_results(round_candidates))

    def _pending_expand_ids(self, *, limit: Optional[int] = None) -> list[str]:
        seed_ids: list[str] = []
        for entry in self.paper_pool.get_ranked_papers():
            if entry.expand or entry.expanding or entry.score <= self.expand_threshold:
                continue
            seed_ids.append(entry.paper.arxiv_id)
            if limit is not None and len(seed_ids) >= limit:
                break
        return seed_ids

    def _answer_count(self) -> int:
        return sum(1 for entry in self.paper_pool.papers.values() if entry.score >= self.answer_score_threshold)

    def weakly_relevant_count(self) -> int:
        return sum(1 for entry in self.paper_pool.papers.values() if int(entry.score) == 1)

    def weak_low_medium_count(self) -> int:
        count = 0
        for arxiv_id, entry in self.paper_pool.papers.items():
            judgment = self.paper_judgment_cache.get(arxiv_id)
            if int(entry.score) == 1 and judgment is not None and judgment.confidence in EXTRA_WEAK_CONFIDENCES:
                count += 1
        return count

    def _has_enough_answers(self) -> bool:
        return self.max_answer_count is not None and self._answer_count() >= self.max_answer_count

    def _has_reached_scored_paper_limit(self) -> bool:
        return (
            self.max_scored_paper_count is not None
            and len(self.paper_judgment_cache) >= self.max_scored_paper_count
        )

    def _should_stop(self) -> bool:
        return self._has_enough_answers() or self._has_reached_scored_paper_limit()

    async def expand_once(self, user_query: str, arxiv_id: str) -> tuple[list[str], list[LoggedPaperResult]]:
        base_id = normalize_arxiv_id(arxiv_id)
        paper_pool_entry = self.paper_pool.get_paper(base_id)
        if paper_pool_entry is None or paper_pool_entry.expand:
            return [], []

        paper_pool_entry.expand = True
        paper_pool_entry.expanding = True
        root_paper_id = paper_pool_entry.paper.raw_paper_id or paper_pool_entry.paper.paper_id or paper_pool_entry.paper.arxiv_id

        try:
            citations, references = await asyncio.gather(
                self.paper_client.get_citations(root_paper_id, limit=self.citations_limit),
                self.paper_client.get_references(root_paper_id, limit=self.references_limit),
            )
        except Exception:
            paper_pool_entry.expanding = False
            return [], []

        merged_candidates: list[Paper] = []
        seen_candidate_ids: set[str] = set()
        for candidate in citations + references:
            candidate_key = candidate.raw_paper_id or candidate.paper_id or candidate.arxiv_id
            if not candidate_key or candidate_key == root_paper_id or candidate_key in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_key)
            merged_candidates.append(candidate)

        hydrated_candidates = await asyncio.gather(*(self._hydrate_candidate(candidate) for candidate in merged_candidates))
        loggable_papers = self._normalize_loggable_papers([paper for paper in hydrated_candidates if paper is not None])

        if not loggable_papers:
            paper_pool_entry.expanding = False
            return [], []

        scored_papers, judgments = await self.score_loggable_papers(user_query, loggable_papers)
        logged_candidates = self._build_logged_results(scored_papers, judgments)

        judgment_by_arxiv_id = {paper.arxiv_id: judgment for paper, judgment in zip(scored_papers, judgments)}
        candidate_papers = self._dedupe_and_normalize_candidates(scored_papers)
        new_ids: list[str] = []
        for paper in candidate_papers:
            score = float(judgment_by_arxiv_id[paper.arxiv_id].relevance_level)
            if score <= self.selector_threshold or self.paper_pool.has_paper(paper.arxiv_id):
                continue
            origin = f"[{paper_pool_entry.paper.arxiv_id}] {paper_pool_entry.paper.title}"
            self.paper_pool.add_paper(paper, "expand", origin, score)
            new_ids.append(paper.arxiv_id)
            if self._should_stop():
                break

        paper_pool_entry.expanding = False
        return new_ids, logged_candidates

    async def score_loggable_papers(self, user_query: str, papers: list[Paper]) -> tuple[list[Paper], list[PaperJudgment]]:
        scored_by_arxiv_id: dict[str, tuple[Paper, PaperJudgment]] = {}
        papers_to_score: list[Paper] = []
        remaining_score_slots = (
            None
            if self.max_scored_paper_count is None
            else max(self.max_scored_paper_count - len(self.paper_judgment_cache), 0)
        )

        for paper in papers:
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            if not arxiv_id:
                continue
            cached_judgment = self.paper_judgment_cache.get(arxiv_id)
            if cached_judgment is not None:
                self.duplicate_scored_paper_ids.add(arxiv_id)
                if arxiv_id in self.prefilter_rejected_ids:
                    continue
                scored_by_arxiv_id[arxiv_id] = (paper, cached_judgment)
                continue

            pool_entry = self.paper_pool.get_paper(arxiv_id)
            if pool_entry is not None:
                self.duplicate_scored_paper_ids.add(arxiv_id)
                cached_judgment = self.paper_judgment_cache.get(arxiv_id)
                if cached_judgment is None:
                    cached_judgment = PaperJudgment(
                        relevance_level=int(pool_entry.score),
                        confidence="medium",
                    )
                    self.paper_judgment_cache[arxiv_id] = cached_judgment
                scored_by_arxiv_id[arxiv_id] = (paper, cached_judgment)
                continue

            if remaining_score_slots is not None:
                if remaining_score_slots <= 0:
                    continue
                remaining_score_slots -= 1
            papers_to_score.append(paper)

        prefilter_scores = await asyncio.gather(
            *(self.get_prefilter_score(user_query, paper) for paper in papers_to_score),
            return_exceptions=True,
        )
        papers_for_judge: list[Paper] = []
        for paper, raw_score in zip(papers_to_score, prefilter_scores):
            if isinstance(raw_score, BaseException):
                papers_for_judge.append(paper)
                continue
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            if not arxiv_id:
                continue
            if float(raw_score) < self.scorer_prefilter_threshold:
                self.paper_judgment_cache[arxiv_id] = PaperJudgment(
                    relevance_level=0,
                    confidence="high",
                )
                self.prefilter_rejected_ids.add(arxiv_id)
                continue
            self.prefilter_passed_ids.add(arxiv_id)
            papers_for_judge.append(paper)

        raw_judgments = await asyncio.gather(
            *(self.get_relevance_judgment(user_query, paper) for paper in papers_for_judge),
            return_exceptions=True,
        )
        for paper, raw_judgment in zip(papers_for_judge, raw_judgments):
            if isinstance(raw_judgment, BaseException):
                continue
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            if not arxiv_id:
                continue
            self.paper_judgment_cache[arxiv_id] = raw_judgment
            scored_by_arxiv_id[arxiv_id] = (paper, raw_judgment)

        scored_papers: list[Paper] = []
        judgments: list[PaperJudgment] = []
        for paper in papers:
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            scored = scored_by_arxiv_id.get(arxiv_id)
            if scored is None:
                continue
            scored_paper, judgment = scored
            scored_papers.append(scored_paper)
            judgments.append(judgment)
        return scored_papers, judgments

    async def _hydrate_candidate(self, candidate: Paper) -> Optional[Paper]:
        target_id = candidate.raw_paper_id or candidate.paper_id or candidate.arxiv_id
        paper = candidate
        if not paper.abstract and target_id:
            try:
                detail = await self.paper_client.get_paper(target_id)
            except Exception:
                detail = None
            if detail is not None:
                paper = detail

        normalized_id = normalize_arxiv_id(paper.arxiv_id)
        if not normalized_id or not paper.abstract:
            return None

        try:
            return paper.model_copy(update={"arxiv_id": normalized_id, "paper_id": normalized_id or paper.paper_id})
        except AttributeError:
            return paper.copy(update={"arxiv_id": normalized_id, "paper_id": normalized_id or paper.paper_id})

    def _dedupe_and_normalize_candidates(self, papers: Iterable[Paper]) -> list[Paper]:
        normalized_papers: list[Paper] = []
        seen_base_ids: set[str] = set()

        for paper in papers:
            base_id = normalize_arxiv_id(paper.arxiv_id)
            if not base_id or base_id in seen_base_ids or self.paper_pool.has_paper(base_id):
                continue
            seen_base_ids.add(base_id)
            try:
                normalized = paper.model_copy(update={"arxiv_id": base_id, "paper_id": base_id or paper.paper_id})
            except AttributeError:
                normalized = paper.copy(update={"arxiv_id": base_id, "paper_id": base_id or paper.paper_id})
            normalized_papers.append(normalized)
        return normalized_papers

    def _normalize_loggable_papers(self, papers: Iterable[Paper]) -> list[Paper]:
        normalized_papers: list[Paper] = []
        seen_base_ids: set[str] = set()

        for paper in papers:
            base_id = normalize_arxiv_id(paper.arxiv_id)
            if not base_id or base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)
            try:
                normalized = paper.model_copy(update={"arxiv_id": base_id, "paper_id": base_id or paper.paper_id})
            except AttributeError:
                normalized = paper.copy(update={"arxiv_id": base_id, "paper_id": base_id or paper.paper_id})
            normalized_papers.append(normalized)
        return normalized_papers

    def _build_logged_results(self, papers: Iterable[Paper], judgments: Iterable[PaperJudgment]) -> list[LoggedPaperResult]:
        return [
            LoggedPaperResult(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                score=float(judgment.relevance_level),
                relevance_level=judgment.relevance_level,
                confidence=judgment.confidence,
            )
            for paper, judgment in zip(papers, judgments)
        ]

    def _dedupe_logged_results(self, papers: Iterable[LoggedPaperResult]) -> list[LoggedPaperResult]:
        deduped: dict[str, LoggedPaperResult] = {}
        for paper in papers:
            if not paper.arxiv_id:
                continue
            previous = deduped.get(paper.arxiv_id)
            if previous is None or paper.score > previous.score:
                deduped[paper.arxiv_id] = paper
        return list(deduped.values())

    def _append_interaction_log(self, source: str, papers: list[LoggedPaperResult]) -> None:
        self.interaction_logs.append(InteractionLogResult(source=source, papers=papers))

    async def get_prefilter_score(self, user_query: str, paper: Paper) -> float:
        prompt = (
            "You are an elite researcher in AI. Evaluate whether the searched paper is relevant to the user query.\n\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n\n"
            f"User Query: {user_query}"
        )
        payload = {"model": self.selector_model_name, "input": [prompt]}
        last_exc: Optional[BaseException] = None
        attempted_urls: set[str] = set()

        for _ in range(len(self.scorer_pool)):
            backend = self.scorer_pool.reserve(excluded=attempted_urls)
            attempted_urls.add(backend["url"])
            try:
                response = await httpx_request_with_retry(
                    self.scorer_client,
                    "POST",
                    backend["url"],
                    json=payload,
                    max_retries=3,
                )
                response.raise_for_status()
                body: Any = response.json()
                predictions = body["data"]
                return float(predictions[0]["probs"][0])
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No scorer backend configured")

    async def get_relevance_judgment(self, user_query: str, paper: Paper) -> PaperJudgment:
        arxiv_id = normalize_arxiv_id(paper.arxiv_id)
        if arxiv_id:
            cached_judgment = self.paper_judgment_cache.get(arxiv_id)
            if cached_judgment is not None:
                return cached_judgment

        user_prompt = build_judge_user_prompt(
            query=user_query,
            title=paper.title,
            abstract=paper.abstract,
        )
        last_exc: Optional[BaseException] = None
        attempted_urls: set[str] = set()

        for _ in range(len(self.judge_pool)):
            backend = self.judge_pool.reserve(excluded=attempted_urls)
            attempted_urls.add(backend["url"])
            try:
                response = await asyncio.to_thread(
                    call_openai_chat,
                    model_name=self.judge_model,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    api_key=self.judge_api_key,
                    api_base=backend["url"],
                )
                message_content = response.choices[0].message.content or ""
                judgment = self._parse_relevance_judgment(message_content)
                if arxiv_id:
                    self.paper_judgment_cache[arxiv_id] = judgment
                return judgment
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No judge backend configured")

    def _parse_relevance_judgment(self, raw_content: str) -> PaperJudgment:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()

        payload: Any = json.loads(self._extract_json_object(content))
        if not isinstance(payload, dict):
            raise ValueError("Judge response must be a JSON object.")

        relevance_level = payload.get("relevance_level")
        if isinstance(relevance_level, str) and relevance_level.strip().isdigit():
            relevance_level = int(relevance_level.strip())
        if not isinstance(relevance_level, int) or relevance_level not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid relevance_level in judge response: {relevance_level!r}")

        confidence = str(payload.get("confidence") or "").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        return PaperJudgment(
            relevance_level=relevance_level,
            confidence=confidence,
        )

    def _extract_json_object(self, content: str) -> str:
        try:
            json.loads(content)
            return content
        except Exception:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge response does not contain a JSON object.")
        return content[start : end + 1]

    def build_second_round_context(self, query: str, first_round_queries: list[str]) -> str:
        ranked_entries = self.paper_pool.get_ranked_papers()[: self.context_paper_count]
        context_lines = [f"Original query: {query}", "", "First round queries:"]
        context_lines.extend(f"- {item}" for item in first_round_queries)
        context_lines.extend(["", "Retrieved papers:"])
        for index, entry in enumerate(ranked_entries, start=1):
            context_lines.append(f"{index}. Title: {entry.paper.title}")
            context_lines.append(f"   Abstract: {entry.paper.abstract}")
        return "\n".join(context_lines)

    def export_ranked_results(self) -> list[RankedPaperResult]:
        results: list[RankedPaperResult] = []
        for entry in self.paper_pool.get_ranked_papers():
            judgment = self.paper_judgment_cache.get(
                entry.paper.arxiv_id,
                PaperJudgment(relevance_level=int(entry.score), confidence="medium"),
            )
            results.append(
                RankedPaperResult(
                    arxiv_id=entry.paper.arxiv_id,
                    title=entry.paper.title,
                    abstract=entry.paper.abstract,
                    score=entry.score,
                    relevance_level=int(entry.score),
                    confidence=judgment.confidence,
                )
            )
        return results
