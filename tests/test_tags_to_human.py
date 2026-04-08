from __future__ import annotations

import pytest

from src.tags.tags_to_human import HumanRenderer, TagPrediction


@pytest.fixture
def renderer() -> HumanRenderer:
    return HumanRenderer()


def test_novelty_flag_overrides_everything(renderer: HumanRenderer) -> None:
    preds = [
        TagPrediction("contactCall", 0.99, "functional"),
        TagPrediction("highArousalLikely", 0.9, "acoustic"),
    ]
    out = renderer.render(preds, novelty_flag=True)
    assert out == "Novel pattern relative to the reference library; interpret cautiously."


def test_unknown_function_message(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("contactCall", 0.9, "functional", unknown_function=True)]
    out = renderer.render(preds, novelty_flag=False)
    assert out == "Unknown vocalization: function could not be classified reliably."


def test_empty_predictions_no_novelty(renderer: HumanRenderer) -> None:
    out = renderer.render([], novelty_flag=False)
    assert out == "No confident tag predictions to report."


def test_high_confidence_functional(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("contactCall", 0.9, "functional")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Functional interpretation:" in out
    assert "functions as contact call" in out


def test_medium_confidence_functional(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("contactCall", 0.65, "functional")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Functional interpretation:" in out
    assert "likely functions as contact call" in out


def test_low_confidence_only_functional(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("contactCall", 0.3, "functional")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Functional interpretation:" in out
    assert "may relate to contact call" in out


def test_high_confidence_acoustic(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("highArousalLikely", 0.85, "acoustic")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Acoustic observations:" in out
    assert "aligns with high arousal" in out


def test_medium_confidence_acoustic(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("highArousalLikely", 0.6, "acoustic")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Acoustic observations:" in out
    assert "likely aligns with high arousal" in out


def test_low_confidence_only_acoustic(renderer: HumanRenderer) -> None:
    preds = [TagPrediction("highArousalLikely", 0.2, "acoustic")]
    out = renderer.render(preds, novelty_flag=False)
    assert "Acoustic observations:" in out
    assert "weak alignment with high arousal" in out


def test_mixed_acoustic_and_functional(renderer: HumanRenderer) -> None:
    preds = [
        TagPrediction("highArousalLikely", 0.85, "acoustic"),
        TagPrediction("contactCall", 0.9, "functional"),
    ]
    out = renderer.render(preds, novelty_flag=False)
    assert "Acoustic observations:" in out
    assert "Functional interpretation:" in out
    acoustic_pos = out.index("Acoustic observations:")
    functional_pos = out.index("Functional interpretation:")
    assert acoustic_pos < functional_pos


def test_low_ignored_when_high_exists(renderer: HumanRenderer) -> None:
    preds = [
        TagPrediction("contactCall", 0.9, "functional"),
        TagPrediction("flockLocalization", 0.2, "functional"),
    ]
    out = renderer.render(preds, novelty_flag=False)
    assert "contact call" in out
    assert "flock localization" not in out
    assert "flockLocalization" not in out


def test_dict_input_accepted(renderer: HumanRenderer) -> None:
    preds = [
        {
            "tag": "contactCall",
            "confidence": 0.9,
            "kind": "functional",
            "unknown_function": False,
        },
        {
            "tag": "highArousalLikely",
            "confidence": 0.85,
            "kind": "acoustic",
            "unknown_function": False,
        },
    ]
    out = renderer.render(preds, novelty_flag=False)
    assert "functions as contact call" in out
    assert "aligns with high arousal" in out


def test_novelty_with_empty_preds(renderer: HumanRenderer) -> None:
    out = renderer.render([], novelty_flag=True)
    assert out == "Novel pattern relative to the reference library; interpret cautiously."


def test_unknown_function_priority_over_predictions(renderer: HumanRenderer) -> None:
    preds = [
        TagPrediction("contactCall", 0.99, "functional", unknown_function=True),
        TagPrediction("highArousalLikely", 0.95, "acoustic"),
    ]
    out = renderer.render(preds, novelty_flag=False)
    assert out == "Unknown vocalization: function could not be classified reliably."
    assert "contact call" not in out
