from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RewriteQueriesResponse:
    queries: list[str]


@dataclass(slots=True)
class PaperJudgment:
    relevance_level: int
    confidence: str


@dataclass(slots=True)
class LoggedPaperResult:
    arxiv_id: str
    title: str
    score: float
    relevance_level: int
    confidence: str


@dataclass(slots=True)
class InteractionLogResult:
    source: str
    papers: list[LoggedPaperResult] = field(default_factory=list)


@dataclass(slots=True)
class RankedPaperResult:
    arxiv_id: str
    title: str
    abstract: str
    score: float
    relevance_level: int
    confidence: str


@dataclass(slots=True)
class AnswerFindingResult:
    query: str
    first_round_queries: list[str] = field(default_factory=list)
    second_round_queries: list[str] = field(default_factory=list)
    scored_paper_count: int = 0
    scorer_checked_count: int = 0
    prefilter_rejected_count: int = 0
    prefilter_passed_count: int = 0
    duplicate_scored_paper_count: int = 0
    weakly_relevant_count: int = 0
    weak_low_medium_count: int = 0
    logs: list[InteractionLogResult] = field(default_factory=list)
    papers: list[RankedPaperResult] = field(default_factory=list)
