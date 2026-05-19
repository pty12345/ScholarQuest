from __future__ import annotations

import json
from typing import Any


STRICT_BATCH_SYSTEM_PROMPT = """You are a strict relevance judge for academic paper filtering.

Your task is to evaluate whether each candidate paper strictly matches the same user query.
Judge based on the user query, paper title, and paper abstract.

Scoring rules:
- 2 = Strict match. The paper fully matches every core requirement in the query, including the target topic, method/task, setting, constraints, and any specified conditions. All important points in the query must be clearly supported by the title or abstract.
- 1 = Partial match or missing/violated constraint. The paper is related to part of the query, but at least one core requirement is missing, unclear, too broad, or contradicted. If a query constraint is not explicitly matched, assign 1 rather than 2.
- 0 = Mismatch. The paper does not match the query, or the main subject/intent differs from the query.

Confidence:
- high = The title and abstract provide enough evidence for the decision.
- medium = The evidence is somewhat incomplete but the decision is still reasonably supported.
- low = The decision is uncertain due to limited or ambiguous information.

Be conservative. Assign 2 only when the match is explicit and complete. Do not infer missing constraints from vague similarity.

Output valid JSON only:
{
  "results": [
    {
      "paper_index": 1,
      "arxiv_id": "...",
      "reason": "brief English reason based on the query, title, and abstract",
      "strict_score": 0,
      "confidence": "low"
    }
  ]
}

Return exactly one result for each input paper. Preserve the input paper_index and arxiv_id.
Within each result, place "reason" before "strict_score", and place "confidence" last.
Do not output markdown or extra text."""


STRICT_BATCH_USER_PROMPT_TEMPLATE = """USER QUERY:
{query}

PAPER CANDIDATES JSON:
{papers_json}
"""


def build_strict_batch_user_prompt(*, query: str, papers: list[dict[str, Any]]) -> str:
    return STRICT_BATCH_USER_PROMPT_TEMPLATE.format(
        query=query,
        papers_json=json.dumps(papers, ensure_ascii=True, indent=2),
    )
