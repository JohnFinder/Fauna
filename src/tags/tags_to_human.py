from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    confidence: float
    kind: Literal["acoustic", "functional"] = "functional"
    unknown_function: bool = False


def _as_predictions(tag_predictions: list[Any]) -> list[TagPrediction]:
    out: list[TagPrediction] = []
    for p in tag_predictions:
        if isinstance(p, TagPrediction):
            out.append(p)
            continue
        if isinstance(p, dict):
            out.append(
                TagPrediction(
                    tag=str(p["tag"]),
                    confidence=float(p["confidence"]),
                    kind=p.get("kind", "functional"),
                    unknown_function=bool(p.get("unknown_function", False)),
                )
            )
            continue
        raise TypeError("Each prediction must be TagPrediction or dict with tag and confidence.")
    return out


_LABELS: dict[str, str] = {
    "contactCall": "contact call",
    "socialContactMaintenance": "social contact maintenance",
    "flockLocalization": "flock localization",
    "isolationOrSeparationContext": "isolation or separation context",
    "pairBondContextLikely": "pair-bond context",
    "highArousalLikely": "high arousal",
    "socialApproachLikely": "social approach",
}


def _functional_sentence(tag: str, bucket: Literal["high", "medium", "low"]) -> str:
    label = _LABELS.get(tag, tag.replace("_", " "))
    if bucket == "high":
        return f"This vocalization functions as {label}."
    if bucket == "medium":
        return f"This vocalization likely functions as {label}."
    return f"This vocalization may relate to {label}."


def _acoustic_sentence(tag: str, bucket: Literal["high", "medium", "low"]) -> str:
    label = _LABELS.get(tag, tag.replace("_", " "))
    if bucket == "high":
        return f"Acoustically, the call aligns with {label}."
    if bucket == "medium":
        return f"Acoustically, the call likely aligns with {label}."
    return f"Acoustically, there is weak alignment with {label}."


class HumanRenderer:
    def render(self, tag_predictions: list[Any], novelty_flag: bool) -> str:
        preds = _as_predictions(tag_predictions)
        if novelty_flag:
            return "Novel pattern relative to the reference library; interpret cautiously."

        if any(p.unknown_function for p in preds):
            return "Unknown vocalization: function could not be classified reliably."

        if not preds:
            return "No confident tag predictions to report."

        by_kind: dict[str, list[TagPrediction]] = {"acoustic": [], "functional": []}
        for p in preds:
            k = p.kind if p.kind in ("acoustic", "functional") else "functional"
            by_kind[k].append(p)

        parts: list[str] = []

        def render_group(kind: Literal["acoustic", "functional"], items: list[TagPrediction]) -> None:
            if not items:
                return
            high = [p for p in items if p.confidence >= 0.8]
            med = [p for p in items if 0.5 <= p.confidence < 0.8]
            low = [p for p in items if p.confidence < 0.5]

            sentences: list[str] = []
            for p in high:
                if kind == "acoustic":
                    sentences.append(_acoustic_sentence(p.tag, "high"))
                else:
                    sentences.append(_functional_sentence(p.tag, "high"))
            for p in med:
                if kind == "acoustic":
                    sentences.append(_acoustic_sentence(p.tag, "medium"))
                else:
                    sentences.append(_functional_sentence(p.tag, "medium"))
            if not sentences and low:
                p = max(low, key=lambda x: x.confidence)
                if kind == "acoustic":
                    sentences.append(_acoustic_sentence(p.tag, "low"))
                else:
                    sentences.append(_functional_sentence(p.tag, "low"))

            if sentences:
                header = (
                    "Acoustic observations:"
                    if kind == "acoustic"
                    else "Functional interpretation:"
                )
                parts.append(header + " " + " ".join(sentences))

        render_group("acoustic", by_kind["acoustic"])
        render_group("functional", by_kind["functional"])

        if not parts:
            return "No confident tag predictions to report."

        return " ".join(parts)
