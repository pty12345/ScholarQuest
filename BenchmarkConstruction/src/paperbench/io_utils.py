from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path, model_type: type[T]) -> list[T]:
    if not path.exists():
        return []
    records: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(cast(T, cast(Any, model_type).model_validate_json(line)))
    return records


def read_model_list(path: Path, model_type: type[T]) -> list[T]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path, model_type)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("seeds")
        if not isinstance(raw_items, list):
            raise ValueError(f"Expected a list or a mapping with 'seeds' in {path}")
        items = raw_items
    else:
        raise ValueError(f"Expected JSON array or object in {path}")

    return [cast(T, cast(Any, model_type)(**item)) for item in items]


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(cast(str, cast(Any, record).model_dump_json()))
            handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")
