from __future__ import annotations

from enum import Enum


class QueryType(str, Enum):
    SPECIFIC_TOPIC = "specific_topic"
    CROSS_TOPIC = "cross_topic"
    METHOD_ARCHITECTURE = "method_architecture"
    CAPABILITY_APPLICATION = "capability_application"
    COLLECTION_SCOPING = "collection_scoping"
    ANALYSIS_CLAIM = "analysis_claim"


class Scope(str, Enum):
    NARROW = "narrow"
    MEDIUM = "medium"
    BROAD = "broad"


class TerminologyExplicitness(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class RecallRequirement(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceType(str, Enum):
    TOPIC_TO_QUERY = "topic_to_query"
    DIFFICULTY_REWRITE = "difficulty_rewrite"
    BLINDSPOT_SUPPLEMENT = "blindspot_supplement"


class HardFilterStatus(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    REVIEW = "review"


class DifficultyLabel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AdjudicationStatus(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class DedupVerdict(str, Enum):
    DUPLICATE = "duplicate"
    DIFFERENT = "different"
    UNCERTAIN = "uncertain"
