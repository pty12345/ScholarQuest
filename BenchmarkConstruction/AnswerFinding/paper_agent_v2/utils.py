import asyncio
import os
import re
import threading
from functools import total_ordering
from typing import Any, Optional

import httpx
from openai import OpenAI
from pydantic import BaseModel
from sortedcontainers import SortedList

try:
    from .http_retry import httpx_request_with_retry
except ImportError:
    from http_retry import httpx_request_with_retry

DEFAULT_PAPER_FIELDS = "title,abstract,year,authors,externalIds"
DEFAULT_PAPER_SEARCH_V2_BASE_URL = "http://172.16.100.204:4000"
DEFAULT_PAPER_SEARCH_V2_API_KEY = "lw-d7ea4e41519dc1cd03b322d0faa8fb9b"
SEARCH_RANGE_FROM_PARAM = os.getenv("PAPER_SEARCH_V2_RANGE_FROM_PARAM", "from")
SEARCH_RANGE_TO_PARAM = os.getenv("PAPER_SEARCH_V2_RANGE_TO_PARAM", "to")


class ApiKeyPool:
    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.current_index = 0
        self._lock = threading.Lock()

    def get_next_key(self) -> Optional[str]:
        with self._lock:
            if not self.keys:
                return None
            key = self.keys[self.current_index % len(self.keys)]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key

    def remove_key(self, key: str) -> None:
        with self._lock:
            if key in self.keys:
                self.keys.remove(key)
                if self.keys:
                    self.current_index = self.current_index % len(self.keys)
                else:
                    self.current_index = 0


GOOGLE_KEYS = [
    "43518041099660a2575a863feb50b0f945f9cf8e",
    "e855c371aaeb1d1318ce28dd7ca12dc643c011e6",
    "dfddf51e7b4eda349fd966b6786585180cb1ae51",
    "7dd8e1ebcc5caa7af1401cc188a614db0adbb2d3",
    "73381556152daadd9cf49f4dfdfe3d6aca9f1804",
    "572f22bd262c4f29f8da80e601526226632aa937",
    "ea3161476f70afdcc2c36e180b35b7ba92305c03",
    "3d3f23dd4044caaf381c08b039c7933f071f8b98",
    "2323273e281a2a3a76d173fa9ac8c105728d4360",
    "5b804c43d17564a7688ca9799ad4e14ccba25a56",
    "f6b0a4dd35bc089a1491d634d1c1f5e578e4b4a6",
    "7e5fa1ff9b55f93a49d567906215b9e8a0e08244",
    "917b4b3977317ab6803917fc693d7bb2578c8f69",
    "21684c929292fbf81142dbb88b35443850654cff",
    "a5205f5e1d47fc0006bd06c620a84af560abe571",
    "6dbec7f212f4676a95186c3b2fb9e7a4fc7f85e8",
    "ed6a22d574158595bfbab57dbd4bab37fabbb3b8",
    "ec2e36c3b9445cbf03fdbc92f725a2465742bebd",
    "522b2e8777db125279e957f07bb95dc6ed7e5c87",
    "2a9a23b1bf4a5c236946f784c22f568ded363c91",
    "dd2e0875e6f4a233b8cea6bf62ab3d480a5097fb",
    "ed6a22d574158595bfbab57dbd4bab37fabbb3b8",
    "ec2e36c3b9445cbf03fdbc92f725a2465742bebd",
    "522b2e8777db125279e957f07bb95dc6ed7e5c87",
    "2a9a23b1bf4a5c236946f784c22f568ded363c91",
    "dd2e0875e6f4a233b8cea6bf62ab3d480a5097fb",
]
google_key_pool = ApiKeyPool(GOOGLE_KEYS)


def normalize_arxiv_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    matched = re.match(r"^(.*)v\d+$", arxiv_id.strip())
    return matched.group(1) if matched else arxiv_id.strip()


def _format_authors(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        names: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    names.append(str(name))
            elif author:
                names.append(str(author))
        return ", ".join(names)
    return str(authors)


class Paper(BaseModel):
    paper_id: str = ""
    raw_paper_id: str = ""
    arxiv_id: str = ""
    title: str
    abstract: str
    authors: str = ""
    year: Optional[int] = None
    score: float = 0.0


@total_ordering
class PaperPoolEntry(BaseModel):
    paper: Paper
    source: str
    origin: str
    score: float
    expand: bool = False
    expanding: bool = False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        return (self.score, self.paper.arxiv_id) < (other.score, other.paper.arxiv_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PaperPoolEntry):
            return NotImplemented
        return self.score == other.score and self.paper.arxiv_id == other.paper.arxiv_id

    def __hash__(self) -> int:
        return hash(self.paper.arxiv_id)


class PaperPool:
    def __init__(self, max_size: int = 20, threshold: float = 0.0, max_abstract_words: int = 400):
        self.papers: dict[str, PaperPoolEntry] = {}
        self.order_papers: list[str] = []
        self.ranked_papers: SortedList[PaperPoolEntry] = SortedList()
        self.max_size = max_size
        self.threshold = threshold
        self.max_abstract_words = max_abstract_words

    def add_paper(self, paper: Paper, source: str, origin: str, score: float) -> None:
        if not paper.arxiv_id or paper.arxiv_id in self.papers:
            return
        entry = PaperPoolEntry(paper=paper, source=source, origin=origin, score=score)
        self.papers[paper.arxiv_id] = entry
        self.ranked_papers.add(entry)
        self.order_papers.append(paper.arxiv_id)

    def get_paper(self, arxiv_id: str) -> Optional[PaperPoolEntry]:
        return self.papers.get(arxiv_id)

    def has_paper(self, arxiv_id: str) -> bool:
        return arxiv_id in self.papers

    def get_order_papers(self) -> list[str]:
        return self.order_papers

    def get_ranked_papers(self) -> list[PaperPoolEntry]:
        return list(self.ranked_papers)[::-1]

    def __len__(self) -> int:
        return len([entry for entry in self.papers.values() if entry.score >= 0.01])

    @property
    def paper_list(self) -> str:
        if not self.papers:
            return "No papers in the pool."

        expanded = [entry for entry in self.ranked_papers if entry.expand and entry.score >= self.threshold][::-1]
        unexpanded = [entry for entry in self.ranked_papers if not entry.expand and entry.score >= self.threshold][::-1]

        half = self.max_size // 2
        display = sorted(expanded[:half] + unexpanded[:half], key=lambda item: item.score, reverse=True)
        if not display:
            return "No relevant papers found above threshold."

        lines = ["Paper Pool Status:", "[EXP]: Expanded, [NEW]: New", "Format: [id] (score) [STATUS] Title", ""]
        for entry in display:
            status = "[EXP]" if entry.expand else "[NEW]"
            abstract_words = entry.paper.abstract.split()
            abstract = " ".join(abstract_words[: self.max_abstract_words])
            if len(abstract_words) > self.max_abstract_words:
                abstract += "..."
            lines.append(
                f"[{entry.paper.arxiv_id}] ({entry.score:.2f}) {status} {entry.paper.title}\n"
                f"Abstract: {abstract}"
            )
        return "\n\n".join(lines)


class PaperSearchV2Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        *,
        max_concurrency: Optional[int] = 16,
        max_detail_concurrency: Optional[int] = 16,
    ):
        self.base_url = (
            base_url or os.getenv("PAPER_SEARCH_V2_BASE_URL", DEFAULT_PAPER_SEARCH_V2_BASE_URL)
        ).rstrip("/")
        self.paper_search_api_key = os.getenv("PAPER_SEARCH_V2_API_KEY", DEFAULT_PAPER_SEARCH_V2_API_KEY).strip()
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
        )
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
        self._detail_semaphore = (
            asyncio.Semaphore(max_detail_concurrency) if max_detail_concurrency and max_detail_concurrency > 0 else None
        )

    async def _request(
        self, method: str, url: str, *, semaphore: Optional[asyncio.Semaphore] = None, **kwargs: Any
    ) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-API-Key"] = self.paper_search_api_key
        effective_semaphore = self._semaphore if semaphore is None else semaphore
        return await httpx_request_with_retry(
            self.client,
            method,
            url,
            semaphore=effective_semaphore,
            headers=headers,
            **kwargs,
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _record_to_paper(data: dict[str, Any]) -> Paper:
        raw_paper_id = str(data.get("paperId") or data.get("paper_id") or "")
        external_ids = data.get("externalIds") or {}
        arxiv_id = normalize_arxiv_id(str(data.get("arxiv_id") or external_ids.get("ArXiv") or ""))
        return Paper(
            paper_id=arxiv_id or raw_paper_id,
            raw_paper_id=raw_paper_id,
            arxiv_id=arxiv_id,
            title=str(data.get("title") or ""),
            abstract=str(data.get("abstract") or ""),
            authors=_format_authors(data.get("authors")),
            year=data.get("year"),
            score=float(data.get("score", 0.0) or 0.0),
        )

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data")
        return data if isinstance(data, list) else []

    async def search(
        self,
        query: str,
        limit: int = 10,
        *,
        source: str = "local_db",
        year: Optional[str] = None,
        from_month: Optional[str] = None,
        to_month: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[Paper]:
        if source == "google":
            return await self._google_search(query=query, limit=limit)
        if source != "local_db":
            raise ValueError(f"Invalid source: {source}")

        params: dict[str, Any] = {"query": query, "limit": limit}
        if year:
            params["year"] = year
        if from_month:
            params[SEARCH_RANGE_FROM_PARAM] = from_month
        if to_month:
            params[SEARCH_RANGE_TO_PARAM] = to_month
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if fields:
            params["fields"] = fields

        response = await self._request("GET", "/paper/search", params=params)
        response.raise_for_status()
        payload = response.json()
        return [self._record_to_paper(item) for item in self._extract_items(payload)]

    async def _google_search(self, query: str, limit: int = 10) -> list[Paper]:
        url = "https://google.serper.dev/search"
        payload = {"q": f"{query} site:arxiv.org", "num": min(limit, 10), "page": 1}

        arxiv_ids: list[str] = []
        num_keys = len(google_key_pool.keys)
        for _ in range(max(3, num_keys)):
            key = google_key_pool.get_next_key()
            if not key:
                break

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers={"X-API-KEY": key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=10,
                    )
                if response.status_code == 200:
                    for item in response.json().get("organic", []):
                        matched = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", item.get("link", ""))
                        if matched:
                            arxiv_ids.append(normalize_arxiv_id(matched.group(1)))
                    break
                google_key_pool.remove_key(key)
            except Exception:
                google_key_pool.remove_key(key)

        papers: list[Paper] = []
        seen_ids: set[str] = set()
        for arxiv_id in arxiv_ids:
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            paper = await self.get_paper(arxiv_id)
            if paper is not None:
                papers.append(paper)
        return papers

    async def get_paper(self, paper_id: str, fields: str = DEFAULT_PAPER_FIELDS) -> Optional[Paper]:
        params = {"fields": fields} if fields else None
        response = await self._request("GET", f"/paper/{paper_id}", params=params, semaphore=self._detail_semaphore)
        # Google may surface papers that are not yet indexed by the detail service.
        # Treat missing detail records as a skipped candidate instead of failing the whole search step.
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not data:
            return None
        return self._record_to_paper(data)

    async def get_citations(self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS) -> list[Paper]:
        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields

        response = await self._request("GET", f"/paper/{paper_id}/citations", params=params)
        response.raise_for_status()
        payload = response.json()
        papers: list[Paper] = []
        for item in self._extract_items(payload):
            citing_paper = item.get("citingPaper")
            if isinstance(citing_paper, dict):
                papers.append(self._record_to_paper(citing_paper))
        return papers

    async def get_references(self, paper_id: str, limit: int = 50, fields: str = DEFAULT_PAPER_FIELDS) -> list[Paper]:
        if limit < 0:
            limit = 99

        params: dict[str, Any] = {"limit": limit}
        if fields:
            params["fields"] = fields

        response = await self._request("GET", f"/paper/{paper_id}/references", params=params)
        response.raise_for_status()
        payload = response.json()
        papers: list[Paper] = []
        for item in self._extract_items(payload):
            cited_paper = item.get("citedPaper")
            if isinstance(cited_paper, dict):
                papers.append(self._record_to_paper(cited_paper))
        return papers


def call_openai_chat(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
):
    client = OpenAI(api_key=api_key, base_url=api_base)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)
