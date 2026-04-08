from __future__ import annotations

import pytest

from src.tags.text_to_tags import TextToTagsParser

_CONTACT_TAGS = [
    "contactCall",
    "flockLocalization",
    "socialContactMaintenance",
]

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


@pytest.fixture
def parser() -> TextToTagsParser:
    return TextToTagsParser()


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_string_rejected(parser: TextToTagsParser, text: str) -> None:
    r = parser.parse(text)
    assert r.rejected is True
    assert r.rejection_reason == "Empty input."


def test_too_long_text_rejected(parser: TextToTagsParser) -> None:
    text = "a" * 161
    r = parser.parse(text)
    assert r.rejected is True
    assert r.rejection_reason == (
        "Input is too long or too free-form for this parser."
    )


def test_too_many_words_rejected(parser: TextToTagsParser) -> None:
    words = " ".join(f"w{i}" for i in range(15))
    r = parser.parse(words)
    assert r.rejected is True
    assert r.rejection_reason == (
        "Input is too long or too free-form for this parser."
    )


def test_translate_phrase_rejected(parser: TextToTagsParser) -> None:
    r = parser.parse("translate this sentence")
    assert r.rejected is True
    assert r.rejection_reason == (
        "Generic translation requests are not supported."
    )


@pytest.mark.parametrize("term", _PREDATOR_TERMS)
def test_predator_terms_rejected(parser: TextToTagsParser, term: str) -> None:
    r = parser.parse(term)
    assert r.rejected is True
    assert r.rejection_reason == (
        "Predator or threat-related requests are not supported."
    )


def test_contact_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("where are you")
    assert r.rejected is False
    assert r.tags == _CONTACT_TAGS


def test_hello_keyword(parser: TextToTagsParser) -> None:
    r = parser.parse("hello")
    assert r.rejected is False
    assert r.tags == _CONTACT_TAGS


def test_ping_keyword(parser: TextToTagsParser) -> None:
    r = parser.parse("ping")
    assert r.rejected is False
    assert r.tags == _CONTACT_TAGS


def test_isolation_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("I am alone")
    assert r.rejected is False
    assert r.tags == ["isolationOrSeparationContext"]


def test_separated_keyword(parser: TextToTagsParser) -> None:
    r = parser.parse("separated")
    assert r.rejected is False
    assert r.tags == ["isolationOrSeparationContext"]


def test_pair_bond_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("pair bond")
    assert r.rejected is False
    assert r.tags == ["pairBondContextLikely"]


def test_mate_keyword(parser: TextToTagsParser) -> None:
    r = parser.parse("mate")
    assert r.rejected is False
    assert r.tags == ["pairBondContextLikely"]


def test_arousal_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("excited")
    assert r.rejected is False
    assert r.tags == ["highArousalLikely"]


def test_approach_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("come here")
    assert r.rejected is False
    assert r.tags == ["socialApproachLikely"]


def test_combined_keywords(parser: TextToTagsParser) -> None:
    r = parser.parse("isolated hello")
    assert r.rejected is False
    assert r.tags == [
        "contactCall",
        "flockLocalization",
        "isolationOrSeparationContext",
        "socialContactMaintenance",
    ]


def test_no_match_rejected(parser: TextToTagsParser) -> None:
    r = parser.parse("quantum physics")
    assert r.rejected is True
    assert r.rejection_reason is not None
    assert "No supported" in r.rejection_reason


def test_results_are_sorted(parser: TextToTagsParser) -> None:
    samples = [
        "where are you",
        "isolated hello",
        "excited",
        "pair bond",
    ]
    for s in samples:
        r = parser.parse(s)
        assert not r.rejected
        assert r.tags == sorted(r.tags)


def test_case_insensitive(parser: TextToTagsParser) -> None:
    lower = parser.parse("where are you")
    upper = parser.parse("WHERE ARE YOU")
    assert lower.tags == upper.tags
    assert lower.rejected == upper.rejected


def test_whitespace_normalized(parser: TextToTagsParser) -> None:
    r = parser.parse("  where   are   you  ")
    assert r.rejected is False
    assert r.tags == _CONTACT_TAGS


def test_parse_result_not_rejected_has_no_reason(parser: TextToTagsParser) -> None:
    r = parser.parse("hello")
    assert r.rejected is False
    assert r.rejection_reason is None
