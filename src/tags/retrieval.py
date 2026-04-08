from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CallLibraryEntry:
    audio_path: str
    tags: frozenset[str] | set[str]
    caller_id: str | None = None
    context: str | None = None
    embedding: Sequence[float] | None = None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


class CallRetriever:
    def __init__(self) -> None:
        self._entries: list[CallLibraryEntry] = []

    def add_entry(self, entry: CallLibraryEntry) -> None:
        self._entries.append(entry)

    def retrieve(
        self,
        requested_tags: Sequence[str],
        top_k: int = 5,
        filter_caller: str | None = None,
        filter_context: str | None = None,
    ) -> list[tuple[CallLibraryEntry, float]]:
        req = set(requested_tags)
        scored: list[tuple[CallLibraryEntry, float]] = []

        for entry in self._entries:
            if filter_caller is not None and entry.caller_id != filter_caller:
                continue
            if filter_context is not None and entry.context != filter_context:
                continue
            lib_tags = set(entry.tags)
            score = _jaccard(req, lib_tags)
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k <= 0:
            return []
        return scored[:top_k]
