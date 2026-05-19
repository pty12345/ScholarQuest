from __future__ import annotations

import re


WRAPPER_PATTERNS = [
    r"^can you help me find\s+",
    r"^please\s+",
    r"^i want some papers about\s+",
    r"^looking for\s+",
]


def clean_surface_query(raw_query: str) -> str:
    text = raw_query.strip()
    for pattern in WRAPPER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ?.")
