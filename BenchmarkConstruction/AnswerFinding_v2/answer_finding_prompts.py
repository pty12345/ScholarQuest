FIRST_ROUND_REWRITE_SYSTEM_PROMPT = """You rewrite one PASA-style paper-search query into first-round search anchors.

Requirements:
- Return exactly 10 English queries.
- Keep every query close to the original intent.
- Vary the semantic angle, retrieval scope, terminology, method, setting, task, evidence type, or comparison focus.
- Make the 10 queries suitable as independent first-round search inputs for finding answer papers.
- Keep each query concise and retrieval-oriented.
- Avoid duplicates and near-duplicates.
- Avoid drifting into a different research problem.
- Do not mention citations, references, hops, or tool usage.
- Do not output markdown or explanations.

Return strict JSON only:
{"queries": ["...", "..."]}
"""


FIRST_ROUND_REWRITE_USER_PROMPT = """Original query:
{query}
"""


SECOND_ROUND_REWRITE_SYSTEM_PROMPT = """You rewrite a paper-search query into a new set of diverse English retrieval queries using retrieved paper context.

Requirements:
- Return exactly 10 English queries.
- Use the provided context papers and previous queries to explore nearby but still relevant directions.
- Avoid repeating the previous round queries.
- Keep each query concise and retrieval-oriented.
- Prefer query forms that may improve recall over related methods, settings, tasks, and formulations.
- Do not mention citations, references, hops, or tool usage.
- Do not output markdown or explanations.

Return strict JSON only:
{"queries": ["...", "..."]}
"""


SECOND_ROUND_REWRITE_USER_PROMPT = """Original query:
{query}

Previous round queries:
{previous_queries}

Retrieved paper context:
{paper_context}
"""
