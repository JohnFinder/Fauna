from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Final


class TagLayer(Enum):
    ACOUSTIC = auto()
    FUNCTIONAL = auto()
    CONTEXT = auto()


class EvidenceLevel(Enum):
    SUPPORTED = auto()
    INFERRED = auto()
    DO_NOT_CLAIM = auto()


class EvidenceSource(StrEnum):
    LITERATURE = "literature"
    HUMAN_ANNOTATION = "humanAnnotation"
    MODEL_INFERENCE = "modelInference"
    PLAYBACK_VALIDATED = "playbackValidated"


@dataclass(frozen=True, slots=True)
class TagMetadata:
    name: str
    layer: TagLayer
    evidence_level: EvidenceLevel
    description: str


@dataclass(frozen=True, slots=True)
class TagPrediction:
    tag_name: str
    probability: float
    evidence_source: EvidenceSource


@dataclass(slots=True)
class TagEvent:
    predictions: list[TagPrediction]
    novelty_flag: bool


ONTOLOGY_VERSION: Final[str] = "1.0.0"

_TAG_DEFINITIONS: Final[tuple[TagMetadata, ...]] = (
    TagMetadata(
        "contactCall",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.SUPPORTED,
        "Contact vocalization maintaining flock or pair cohesion.",
    ),
    TagMetadata(
        "socialContactMaintenance",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.SUPPORTED,
        "Vocalization reinforcing an ongoing social bond.",
    ),
    TagMetadata(
        "flockLocalization",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.SUPPORTED,
        "Call supporting spatial localization within the flock.",
    ),
    TagMetadata(
        "isolationOrSeparationContext",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Produced during or after separation from conspecifics.",
    ),
    TagMetadata(
        "sharedCallMatch",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.SUPPORTED,
        "Call matches a partner or flock member's typical pattern.",
    ),
    TagMetadata(
        "individualSignaturePresent",
        TagLayer.ACOUSTIC,
        EvidenceLevel.SUPPORTED,
        "Acoustic cues consistent with an individual signature.",
    ),
    TagMetadata(
        "highModulation",
        TagLayer.ACOUSTIC,
        EvidenceLevel.SUPPORTED,
        "Substantial frequency or amplitude modulation in the segment.",
    ),
    TagMetadata(
        "narrowbandFm",
        TagLayer.ACOUSTIC,
        EvidenceLevel.SUPPORTED,
        "Narrowband frequency-modulated structure typical of contact calls.",
    ),
    TagMetadata(
        "unknownFunction",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.SUPPORTED,
        "Functional role cannot be assigned from available evidence.",
    ),
    TagMetadata(
        "socialApproachLikely",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.INFERRED,
        "Model-inferred motivation consistent with social approach.",
    ),
    TagMetadata(
        "highArousalLikely",
        TagLayer.FUNCTIONAL,
        EvidenceLevel.INFERRED,
        "Model-inferred elevated arousal or urgency.",
    ),
    TagMetadata(
        "pairBondContextLikely",
        TagLayer.CONTEXT,
        EvidenceLevel.INFERRED,
        "Model-inferred bonded-pair social context.",
    ),
    TagMetadata(
        "visiblePartner",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Partner or focal conspecific is within visual range.",
    ),
    TagMetadata(
        "audioOnlyPartner",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Conspecific audible but not reliably visible.",
    ),
    TagMetadata(
        "separation",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Physically separated from flock or bonded partner.",
    ),
    TagMetadata(
        "groupFlight",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Flock engaged in coordinated flight.",
    ),
    TagMetadata(
        "perchedSocial",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Birds perched in social proximity.",
    ),
    TagMetadata(
        "operantTask",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Recorded during an operant conditioning session.",
    ),
    TagMetadata(
        "playbackTrial",
        TagLayer.CONTEXT,
        EvidenceLevel.SUPPORTED,
        "Recorded during a playback experiment.",
    ),
)

TAG_REGISTRY: Final[dict[str, TagMetadata]] = {m.name: m for m in _TAG_DEFINITIONS}


def get_tags_by_layer(layer: TagLayer) -> list[str]:
    return sorted(name for name, meta in TAG_REGISTRY.items() if meta.layer == layer)


def get_tags_by_evidence(level: EvidenceLevel) -> list[str]:
    return sorted(name for name, meta in TAG_REGISTRY.items() if meta.evidence_level == level)
