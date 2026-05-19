import re
import threading
from typing import Any


def parse_backends(
    *,
    raw_urls: str,
    raw_names: str,
    legacy_url: str,
    default_urls: tuple[str, ...],
    default_prefix: str,
) -> list[dict[str, Any]]:
    urls = [item.strip() for item in re.split(r"[\s,]+", raw_urls) if item.strip()]
    if not urls:
        urls = [legacy_url] if legacy_url else list(default_urls)

    names = [item.strip() for item in re.split(r"[\s,]+", raw_names) if item.strip()]
    backends: list[dict[str, Any]] = []
    for index, url in enumerate(urls):
        backends.append(
            {
                "name": names[index] if index < len(names) and names[index] else f"{default_prefix}-{index + 1}",
                "url": url,
                "requests": 0,
            }
        )
    return backends


def clone_backends(backends: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item["name"]),
            "url": str(item["url"]),
            "requests": int(item.get("requests", 0)),
        }
        for item in backends
    ]


class BackendPool:
    def __init__(self, backends: list[dict[str, Any]], *, backend_kind: str) -> None:
        self.backends = backends
        self.backend_kind = backend_kind
        self._lock = threading.Lock()
        self._tiebreak = 0

    def __len__(self) -> int:
        return len(self.backends)

    def first_url(self) -> str:
        return str(self.backends[0]["url"]) if self.backends else ""

    def reserve(self, *, excluded: set[str] | None = None) -> dict[str, Any]:
        excluded = excluded or set()
        with self._lock:
            available = [
                (index, backend)
                for index, backend in enumerate(self.backends)
                if backend["url"] not in excluded
            ]
            if not available:
                raise RuntimeError(f"No {self.backend_kind} backends available")

            min_requests = min(int(backend["requests"]) for _, backend in available)
            candidate_indexes = [
                index for index, backend in available if int(backend["requests"]) == min_requests
            ]
            start = self._tiebreak % len(self.backends)

            selected_index = candidate_indexes[0]
            for offset in range(len(self.backends)):
                probe_index = (start + offset) % len(self.backends)
                if probe_index in candidate_indexes:
                    selected_index = probe_index
                    break

            backend = self.backends[selected_index]
            backend["requests"] = int(backend["requests"]) + 1
            self._tiebreak = (selected_index + 1) % len(self.backends)
            return {
                "name": str(backend["name"]),
                "url": str(backend["url"]),
                "requests": int(backend["requests"]),
            }
