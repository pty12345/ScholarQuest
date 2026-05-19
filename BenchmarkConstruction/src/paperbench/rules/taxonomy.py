from __future__ import annotations

from ..enums import QueryType
from ..types import QueryRecord


QUERY_TYPE_PRIORITY: tuple[QueryType, ...] = (
    QueryType.COLLECTION_SCOPING,
    QueryType.ANALYSIS_CLAIM,
    QueryType.METHOD_ARCHITECTURE,
    QueryType.CAPABILITY_APPLICATION,
    QueryType.CROSS_TOPIC,
    QueryType.SPECIFIC_TOPIC,
)


def evaluate_type_consistency(record: QueryRecord) -> tuple[bool | None, str | None]:
    if record.target_query_type is None or record.query_type is None:
        return None, None
    if record.target_query_type == record.query_type:
        return True, "classified type matches generation target"
    return (
        False,
        f"classified as {record.query_type.value} but generated for {record.target_query_type.value}",
    )
