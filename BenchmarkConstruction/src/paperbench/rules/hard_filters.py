from __future__ import annotations

import re

from ..enums import HardFilterStatus


FACTOID_PATTERNS = [
    r"^who proposed",
    r"^when was .+ published",
]

EXPLANATION_PATTERNS = [
    r"^what is ",
    r"^explain ",
    r"^tell me about ",
]

QUESTION_OR_INSTRUCTION_PATTERNS = [
    r"^what\b",
    r"^how\b",
    r"^which\b",
    r"^find\b",
    r"^identify\b",
    r"^compare\b",
    r"^list\b",
    r"^show\b",
    r"^can you\b",
    r"^help me\b",
]

TEMPORAL_PATTERNS = [
    r"\brecent\b",
    r"\blatest\b",
    r"\bsince\s+20\d{2}\b",
    r"\bafter\s+20\d{2}\b",
    r"\bbefore\s+20\d{2}\b",
    r"\bpost-?20\d{2}\b",
]

DETAIL_SEEKING_PATTERNS = [
    r"\bhow do (they|these|those)\b",
    r"\bperformance\b",
    r"\befficiency\b",
    r"\bmetrics?\b",
    r"\bdatasets? introduced\b",
    r"\bstrengths?\b",
    r"\bweaknesses?\b",
    r"\bcompare to\b",
    r"\bcomparison to\b",
    r"\bversus\b",
    r"\bvs\.?\b",
]

FULLTEXT_DEPENDENCY_PATTERNS = [
    r"\bevaluated on\b",
    r"\btested on\b",
    r"\bon .+ dataset\b",
    r"\bin zero-shot setting\b",
    r"\bin few-shot setting\b",
    r"\boutperforms?\b",
    r"\bsuperior to\b",
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bresult table\b",
    r"\bablation\b",
    r"\bappendix\b",
]

FINEGRAINED_DETAIL_PATTERNS = [
    r"\bloss variant\b",
    r"\bhyperparameter\b",
    r"\blearning rate\b",
    r"\bbatch size\b",
    r"\btraining trick\b",
    r"\bimplementation detail\b",
]

SUBJECTIVE_PATTERNS = [
    r"\bbest\b",
    r"\blatest\b",
    r"\bmost influential\b",
    r"\bmost important\b",
    r"\bmost cited\b",
    r"\btop\b",
    r"\bstate of the art\b",
]


def run_hard_filter(query: str) -> tuple[HardFilterStatus, str | None]:
    normalized = query.strip().lower()
    if len(normalized.split()) < 4:
        return HardFilterStatus.REJECT, "too_short"

    if len(normalized.split()) > 20:
        return HardFilterStatus.REJECT, "too_long"

    for pattern in FACTOID_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "factoid_qa"

    for pattern in EXPLANATION_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "not_paper_search"

    for pattern in QUESTION_OR_INSTRUCTION_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "not_topic_style"

    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "temporal_constraint"

    for pattern in DETAIL_SEEKING_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "detail_seeking"

    for pattern in FULLTEXT_DEPENDENCY_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "fulltext_dependency"

    for pattern in FINEGRAINED_DETAIL_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "finegrained_detail"

    for pattern in SUBJECTIVE_PATTERNS:
        if re.search(pattern, normalized):
            return HardFilterStatus.REJECT, "subjective_qualifier"

    if "my advisor" in normalized or "last week" in normalized:
        return HardFilterStatus.REJECT, "private_context"

    return HardFilterStatus.PASS, None
