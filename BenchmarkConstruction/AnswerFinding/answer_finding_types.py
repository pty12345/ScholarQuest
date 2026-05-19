from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RewriteQueriesResponse:
    queries: list[str]


@dataclass(slots=True)
class LoggedPaperResult:
    arxiv_id: str
    title: str
    score: float


@dataclass(slots=True)
class InteractionLogResult:
    source: str
    papers: list[LoggedPaperResult] = field(default_factory=list)


@dataclass(slots=True)
class RankedPaperResult:
    title: str
    abstract: str
    score: float


@dataclass(slots=True)
class AnswerFindingResult:
    query: str
    first_round_queries: list[str] = field(default_factory=list)
    second_round_queries: list[str] = field(default_factory=list)
    scored_paper_count: int = 0
    logs: list[InteractionLogResult] = field(default_factory=list)
    papers: list[RankedPaperResult] = field(default_factory=list)
