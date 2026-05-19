from __future__ import annotations

import math
from collections import defaultdict

from ..types import QueryRecord


def stratified_sample(
    records: list[QueryRecord],
    *,
    target_size: int,
    type_distribution: dict[str, float],
) -> list[QueryRecord]:
    by_type: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in records:
        key = record.query_type.value if record.query_type is not None else "unknown"
        by_type[key].append(record)

    type_quotas: dict[str, int] = {}
    for qtype, ratio in type_distribution.items():
        type_quotas[qtype] = max(1, math.floor(target_size * ratio))

    sampled: list[QueryRecord] = []
    for qtype, quota in type_quotas.items():
        bucket = by_type.get(qtype, [])
        sampled.extend(bucket[:quota])

    if len(sampled) < target_size:
        selected_ids = {record.query_id for record in sampled}
        remaining = [record for record in records if record.query_id not in selected_ids]
        sampled.extend(remaining[: target_size - len(sampled)])

    return sampled[:target_size]
