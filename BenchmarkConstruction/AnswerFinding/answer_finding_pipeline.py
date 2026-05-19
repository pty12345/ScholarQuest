from __future__ import annotations

import asyncio
import json
import os
from itertools import cycle
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
        RankedPaperResult,
        RewriteQueriesResponse,
    )
    from .paper_agent_v2.http_retry import httpx_request_with_retry
    from .paper_agent_v2.utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id
except ImportError:
    from answer_finding_prompts import (
        FIRST_ROUND_REWRITE_SYSTEM_PROMPT,
        FIRST_ROUND_REWRITE_USER_PROMPT,
        SECOND_ROUND_REWRITE_SYSTEM_PROMPT,
        SECOND_ROUND_REWRITE_USER_PROMPT,
    )
    from answer_finding_types import (
        AnswerFindingResult,
        InteractionLogResult,
        LoggedPaperResult,
        RankedPaperResult,
        RewriteQueriesResponse,
    )
    from paper_agent_v2.http_retry import httpx_request_with_retry
    from paper_agent_v2.utils import Paper, PaperPool, PaperSearchV2Client, call_openai_chat, normalize_arxiv_id


DEFAULT_REWRITE_MODEL = "qwen-max"
DEFAULT_REWRITE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SCORER_URLS = (
    "http://localhost:8993/classify",
    "http://localhost:8996/classify",
)
ANSWER_FINDING_ROOT = Path(__file__).resolve().parent
load_dotenv(ANSWER_FINDING_ROOT / ".env")


class AnswerFindingPipeline:
    def __init__(
        self,
        *,
        queries_per_round: int = 10,
        search_top_k: int = 10,
        selector_threshold: float = 0.05,
        expand_threshold: float = 0.1,
        citations_limit: int = 30,
        references_limit: int = -1,
        context_paper_count: int = 20,
    ) -> None:
        self.queries_per_round = queries_per_round
        self.search_top_k = search_top_k
        self.selector_threshold = selector_threshold
        self.expand_threshold = expand_threshold
        self.citations_limit = citations_limit
        self.references_limit = references_limit
        self.context_paper_count = context_paper_count

        self.paper_pool = PaperPool(max_size=max(20, context_paper_count))
        self.paper_client = PaperSearchV2Client(timeout=30.0)
        self.selector_client = httpx.AsyncClient(timeout=10.0)

        self.rewrite_model = os.getenv("ANSWER_FINDING_REWRITE_MODEL", DEFAULT_REWRITE_MODEL).strip()
        self.rewrite_api_base = os.getenv("ANSWER_FINDING_REWRITE_API_BASE", DEFAULT_REWRITE_API_BASE).strip()
        self.rewrite_api_key = (
            os.getenv("ANSWER_FINDING_REWRITE_API_KEY", "").strip()
            or os.getenv("PAPER_AGENT_V2_WARMUP_API_KEY", "").strip()
        )
        if not self.rewrite_api_key:
            raise ValueError("Missing rewrite API key. Set ANSWER_FINDING_REWRITE_API_KEY or PAPER_AGENT_V2_WARMUP_API_KEY.")

        raw_scorer_urls = (
            os.getenv("ANSWER_FINDING_SCORER_URLS", "").strip()
            or os.getenv("PAPER_AGENT_V2_SCORER_URLS", "").strip()
        )
        if raw_scorer_urls:
            scorer_urls = [item.strip() for item in raw_scorer_urls.replace(",", " ").split() if item.strip()]
        else:
            legacy_scorer = os.getenv("PAPER_AGENT_V2_SELECTOR_URL", "").strip()
            scorer_urls = [legacy_scorer] if legacy_scorer else list(DEFAULT_SCORER_URLS)
        self.scorer_urls = scorer_urls
        self.selector_model_name = (
            os.getenv("ANSWER_FINDING_SELECTOR_MODEL", "").strip()
            or os.getenv("PAPER_AGENT_V2_SELECTOR_MODEL", "selector").strip()
        )
        self._scorer_cycle = cycle(self.scorer_urls)
        self.interaction_logs: list[InteractionLogResult] = []
        self.relevance_score_cache: dict[str, float] = {}

    async def close(self) -> None:
        await self.selector_client.aclose()
        await self.paper_client.close()

    async def run(self, query: str) -> AnswerFindingResult:
        self.interaction_logs = []
        self.relevance_score_cache = {}
        first_round_queries = self.rewrite_queries(query)
        first_round_seed_ids = await self.search_and_score_queries(query, first_round_queries)
        await self.expand_until_converged(query, first_round_seed_ids)

        return AnswerFindingResult(
            query=query,
            first_round_queries=first_round_queries,
            second_round_queries=[],
            scored_paper_count=len(self.relevance_score_cache),
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

            scored_papers, relevance_scores = await self.score_loggable_papers(user_query, loggable_papers)
            self._append_interaction_log(source, self._build_logged_results(scored_papers, relevance_scores))

            score_by_arxiv_id = {paper.arxiv_id: score for paper, score in zip(scored_papers, relevance_scores)}
            candidate_papers = self._dedupe_and_normalize_candidates(scored_papers)
            for paper in candidate_papers:
                score = score_by_arxiv_id[paper.arxiv_id]
                if score <= self.selector_threshold or self.paper_pool.has_paper(paper.arxiv_id):
                    continue
                self.paper_pool.add_paper(paper, "search", query, score)
                seed_ids.append(paper.arxiv_id)
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

    async def expand_until_converged(self, user_query: str, initial_seed_ids: list[str]) -> None:
        await self.expand_papers(user_query, initial_seed_ids, source_label="expand:initial")

        iteration = 1
        while True:
            seed_ids = self._pending_expand_ids()
            if not seed_ids:
                return
            await self.expand_papers(user_query, seed_ids, source_label=f"expand:iter{iteration}")
            iteration += 1

    async def expand_papers(self, user_query: str, seed_ids: list[str], *, source_label: str) -> None:
        round_candidates: list[LoggedPaperResult] = []
        for seed_id in dict.fromkeys(seed_ids):
            _, logged_papers = await self.expand_once(user_query, seed_id)
            round_candidates.extend(logged_papers)
        self._append_interaction_log(source_label, self._dedupe_logged_results(round_candidates))

    def _pending_expand_ids(self) -> list[str]:
        seed_ids: list[str] = []
        for entry in self.paper_pool.get_ranked_papers():
            if entry.expand or entry.expanding or entry.score <= self.expand_threshold:
                continue
            seed_ids.append(entry.paper.arxiv_id)
        return seed_ids

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

        scored_papers, relevance_scores = await self.score_loggable_papers(user_query, loggable_papers)
        logged_candidates = self._build_logged_results(scored_papers, relevance_scores)

        score_by_arxiv_id = {paper.arxiv_id: score for paper, score in zip(scored_papers, relevance_scores)}
        candidate_papers = self._dedupe_and_normalize_candidates(scored_papers)
        new_ids: list[str] = []
        for paper in candidate_papers:
            score = score_by_arxiv_id[paper.arxiv_id]
            if score <= self.selector_threshold or self.paper_pool.has_paper(paper.arxiv_id):
                continue
            origin = f"[{paper_pool_entry.paper.arxiv_id}] {paper_pool_entry.paper.title}"
            self.paper_pool.add_paper(paper, "expand", origin, score)
            new_ids.append(paper.arxiv_id)

        paper_pool_entry.expanding = False
        return new_ids, logged_candidates

    async def score_loggable_papers(self, user_query: str, papers: list[Paper]) -> tuple[list[Paper], list[float]]:
        scored_by_arxiv_id: dict[str, tuple[Paper, float]] = {}
        papers_to_score: list[Paper] = []

        for paper in papers:
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            if not arxiv_id:
                continue
            cached_score = self.relevance_score_cache.get(arxiv_id)
            if cached_score is not None:
                scored_by_arxiv_id[arxiv_id] = (paper, cached_score)
                continue

            pool_entry = self.paper_pool.get_paper(arxiv_id)
            if pool_entry is not None:
                self.relevance_score_cache[arxiv_id] = pool_entry.score
                scored_by_arxiv_id[arxiv_id] = (paper, pool_entry.score)
                continue

            papers_to_score.append(paper)

        raw_scores = await asyncio.gather(
            *(self.get_relevance_score(user_query, paper) for paper in papers_to_score),
            return_exceptions=True,
        )
        for paper, raw_score in zip(papers_to_score, raw_scores):
            if isinstance(raw_score, BaseException):
                continue
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            if not arxiv_id:
                continue
            score = float(raw_score)
            self.relevance_score_cache[arxiv_id] = score
            scored_by_arxiv_id[arxiv_id] = (paper, score)

        scored_papers: list[Paper] = []
        relevance_scores: list[float] = []
        for paper in papers:
            arxiv_id = normalize_arxiv_id(paper.arxiv_id)
            scored = scored_by_arxiv_id.get(arxiv_id)
            if scored is None:
                continue
            scored_paper, score = scored
            scored_papers.append(scored_paper)
            relevance_scores.append(score)
        return scored_papers, relevance_scores

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

    def _build_logged_results(self, papers: Iterable[Paper], scores: Iterable[float]) -> list[LoggedPaperResult]:
        return [
            LoggedPaperResult(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                score=score,
            )
            for paper, score in zip(papers, scores)
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

    async def get_relevance_score(self, user_query: str, paper: Paper) -> float:
        prompt = (
            "You are an elite researcher in AI. Evaluate whether the searched paper is relevant to the user query.\n\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n\n"
            f"User Query: {user_query}"
        )
        payload = {"model": self.selector_model_name, "input": [prompt]}
        last_exc: Optional[BaseException] = None

        for _ in range(len(self.scorer_urls)):
            scorer_url = next(self._scorer_cycle)
            try:
                response = await httpx_request_with_retry(
                    self.selector_client,
                    "POST",
                    scorer_url,
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
            results.append(
                RankedPaperResult(
                    title=entry.paper.title,
                    abstract=entry.paper.abstract,
                    score=entry.score,
                )
            )
        return results
