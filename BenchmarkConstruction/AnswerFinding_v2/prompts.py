SYSTEM_PROMPT = """You are an expert LLM-as-a-Judge for academic paper search.
Your task is to evaluate how relevant a candidate paper is to a given research query.
You must strictly follow the scoring rubric and output only the required JSON.
You should ignore writing quality, length, style, or popularity.
Only evaluate semantic relevance between the query and the paper content.
"""

USER_PROMPT_TEMPLATE = """QUERY:
{query}

PAPER CANDIDATE:
Title: {title}
Abstract: {abstract}

SCORING RUBRIC (relevance_level):
- 3 = Highly relevant — Directly addresses the core research question or problem expressed in the query.
- 2 = Moderately relevant — About a closely related topic, but only partially matches the intent or focuses on a sub-aspect.
- 1 = Weakly relevant — Belongs to the broader area but only tangentially connected to the query.
- 0 = Irrelevant — Not relevant to the query in any meaningful way.

OUTPUT FORMAT (must be valid JSON):
{{
  "confidence": "low/medium/high",
  "relevance_level": 0/1/2/3
}}

You MUST output valid JSON and nothing else. Place relevance_level last.
"""


def build_user_prompt(*, query: str, title: str, abstract: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        title=title,
        abstract=abstract,
    )