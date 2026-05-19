from __future__ import annotations


def make_query_id(index: int) -> str:
    return f"QP_{index:06d}"


def make_cluster_id(index: int) -> str:
    return f"CL_{index:04d}"


def make_duplicate_group_id(index: int) -> str:
    return f"DG_{index:04d}"
