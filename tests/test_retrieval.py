from __future__ import annotations

import pytest

from src.tags.retrieval import CallLibraryEntry, CallRetriever


def test_empty_library_returns_empty() -> None:
    r = CallRetriever()
    assert r.retrieve(["a"], top_k=5) == []


def test_single_entry_exact_match() -> None:
    r = CallRetriever()
    e = CallLibraryEntry("a.wav", frozenset({"x", "y"}))
    r.add_entry(e)
    results = r.retrieve(["x", "y"], top_k=5)
    assert len(results) == 1
    assert results[0][0] is e
    assert results[0][1] == pytest.approx(1.0)


def test_single_entry_partial_match() -> None:
    r = CallRetriever()
    e = CallLibraryEntry("a.wav", frozenset({"a", "b"}))
    r.add_entry(e)
    results = r.retrieve(["a", "c"], top_k=5)
    assert len(results) == 1
    assert results[0][1] == pytest.approx(1.0 / 3.0)


def test_single_entry_no_match() -> None:
    r = CallRetriever()
    e = CallLibraryEntry("a.wav", frozenset({"a", "b"}))
    r.add_entry(e)
    results = r.retrieve(["c", "d"], top_k=5)
    assert len(results) == 1
    assert results[0][1] == pytest.approx(0.0)


def test_ranking_order() -> None:
    r = CallRetriever()
    e_perfect = CallLibraryEntry("p.wav", frozenset({"t1", "t2"}))
    e_half = CallLibraryEntry("h.wav", frozenset({"t1"}))
    e_none = CallLibraryEntry("n.wav", frozenset({"x"}))
    r.add_entry(e_none)
    r.add_entry(e_half)
    r.add_entry(e_perfect)
    results = r.retrieve(["t1", "t2"], top_k=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] is e_perfect
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(1.0 / 2.0)
    assert results[2][1] == pytest.approx(0.0)


def test_top_k_limits_results() -> None:
    r = CallRetriever()
    for i in range(5):
        r.add_entry(CallLibraryEntry(f"{i}.wav", frozenset({f"t{i}"})))
    results = r.retrieve(["t0", "t1", "t2", "t3", "t4"], top_k=2)
    assert len(results) == 2


def test_top_k_zero_returns_empty() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("a.wav", frozenset({"x"})))
    assert r.retrieve(["x"], top_k=0) == []


def test_filter_by_caller() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("a.wav", frozenset({"x"}), caller_id="alice"))
    r.add_entry(CallLibraryEntry("b.wav", frozenset({"x"}), caller_id="bob"))
    results = r.retrieve(["x"], top_k=10, filter_caller="alice")
    assert len(results) == 1
    assert results[0][0].audio_path == "a.wav"


def test_filter_by_context() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("a.wav", frozenset({"x"}), context="nest"))
    r.add_entry(CallLibraryEntry("b.wav", frozenset({"x"}), context="flight"))
    results = r.retrieve(["x"], top_k=10, filter_context="nest")
    assert len(results) == 1
    assert results[0][0].audio_path == "a.wav"


def test_filter_caller_and_context_combined() -> None:
    r = CallRetriever()
    r.add_entry(
        CallLibraryEntry("a.wav", frozenset({"x"}), caller_id="alice", context="nest")
    )
    r.add_entry(
        CallLibraryEntry("b.wav", frozenset({"x"}), caller_id="alice", context="flight")
    )
    r.add_entry(
        CallLibraryEntry("c.wav", frozenset({"x"}), caller_id="bob", context="nest")
    )
    results = r.retrieve(
        ["x"], top_k=10, filter_caller="alice", filter_context="nest"
    )
    assert len(results) == 1
    assert results[0][0].audio_path == "a.wav"


def test_filter_no_match() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("a.wav", frozenset({"x"}), caller_id="alice"))
    assert r.retrieve(["x"], top_k=10, filter_caller="nobody") == []


def test_jaccard_both_empty() -> None:
    r = CallRetriever()
    e = CallLibraryEntry("a.wav", frozenset())
    r.add_entry(e)
    results = r.retrieve([], top_k=5)
    assert len(results) == 1
    assert results[0][1] == pytest.approx(1.0)


def test_jaccard_one_empty() -> None:
    r = CallRetriever()
    e = CallLibraryEntry("a.wav", frozenset({"x"}))
    r.add_entry(e)
    results = r.retrieve([], top_k=5)
    assert len(results) == 1
    assert results[0][1] == pytest.approx(0.0)


def test_multiple_adds() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("1.wav", frozenset({"a"})))
    r.add_entry(CallLibraryEntry("2.wav", frozenset({"a", "b"})))
    assert len(r.retrieve(["a"], top_k=10)) == 2


def test_scores_are_floats() -> None:
    r = CallRetriever()
    r.add_entry(CallLibraryEntry("a.wav", frozenset({"x", "y"})))
    for _, score in r.retrieve(["x"], top_k=5):
        assert type(score) is float
