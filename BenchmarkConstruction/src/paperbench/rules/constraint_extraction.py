from __future__ import annotations


def count_concept_phrases(query: str) -> int:
    """Estimate the number of distinct concept phrases in a query.

    Uses simple heuristics (splitting on conjunctions and prepositions)
    to give a rough concept count for downstream difficulty estimation.
    """
    separators = [" for ", " and ", " with ", " in ", " on ", " using "]
    parts = [query]
    for sep in separators:
        new_parts: list[str] = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    return max(1, sum(1 for p in parts if len(p.strip().split()) >= 1))
