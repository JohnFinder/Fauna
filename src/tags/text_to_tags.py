from __future__ import annotations

import re
from dataclasses import dataclass, field

ONTOLOGY: frozenset[str] = frozenset(
    {
        "contactCall",
        "socialContactMaintenance",
        "flockLocalization",
        "isolationOrSeparationContext",
        "pairBondContextLikely",
        "highArousalLikely",
        "socialApproachLikely",
    }
)

_MAX_CHARS = 160
_MAX_WORDS = 14

_PREDATOR_TERMS = (
    "predator",
    "snake",
    "hawk",
    "attack",
    "eagle",
    "cat",
    "danger",
    "threat",
)

_TRANSLATE_PHRASES = (
    "translate this sentence",
    "translate this",
    "translate the following",
)


@dataclass
class TagParseResult:
    tags: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w])


class TextToTagsParser:
    def __init__(self) -> None:
        self._rules: list[tuple[tuple[str, ...], frozenset[str]]] = [
            (
                (
                    "contact",
                    "hello",
                    "hi",
                    "hey",
                    "greeting",
                    "check in",
                    "checking in",
                    "check-in",
                    "where are you",
                    "where r u",
                    "locate",
                    "localization",
                    "flock",
                    "call out",
                    "reach out",
                    "ping",
                    "hail",
                ),
                frozenset(
                    {
                        "contactCall",
                        "socialContactMaintenance",
                        "flockLocalization",
                    }
                ),
            ),
            (
                (
                    "isolated",
                    "isolation",
                    "alone",
                    "lonely",
                    "separated",
                    "separation",
                    "split up",
                    "lost",
                    "by myself",
                ),
                frozenset({"isolationOrSeparationContext"}),
            ),
            (
                (
                    "pair bond",
                    "pair-bond",
                    "mate",
                    "partner",
                    "spouse",
                    "paired",
                    "bonded pair",
                ),
                frozenset({"pairBondContextLikely"}),
            ),
            (
                (
                    "aroused",
                    "arousal",
                    "excited",
                    "agitated",
                    "worked up",
                    "amped",
                    "hyper",
                    "intense",
                ),
                frozenset({"highArousalLikely"}),
            ),
            (
                (
                    "approach",
                    "come here",
                    "come closer",
                    "come nearer",
                    "move closer",
                    "draw near",
                    "join me",
                ),
                frozenset({"socialApproachLikely"}),
            ),
        ]

    def parse(self, text: str) -> TagParseResult:
        if not text or not text.strip():
            return TagParseResult(
                rejected=True,
                rejection_reason="Empty input.",
            )

        norm = _normalize(text)
        if len(norm) > _MAX_CHARS or _word_count(norm) > _MAX_WORDS:
            return TagParseResult(
                rejected=True,
                rejection_reason="Input is too long or too free-form for this parser.",
            )

        for p in _TRANSLATE_PHRASES:
            if p in norm:
                return TagParseResult(
                    rejected=True,
                    rejection_reason="Generic translation requests are not supported.",
                )

        for term in _PREDATOR_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", norm):
                return TagParseResult(
                    rejected=True,
                    rejection_reason="Predator or threat-related requests are not supported.",
                )

        tags: set[str] = set()
        for phrases, tag_set in self._rules:
            for phrase in phrases:
                if " " in phrase or "-" in phrase:
                    if phrase in norm:
                        tags |= tag_set
                elif re.search(rf"\b{re.escape(phrase)}\b", norm):
                    tags |= tag_set

        valid = sorted(t for t in tags if t in ONTOLOGY)
        if not valid:
            return TagParseResult(
                rejected=True,
                rejection_reason="No supported budgerigar contact-call tags matched the input.",
            )

        return TagParseResult(tags=valid, rejected=False, rejection_reason=None)
