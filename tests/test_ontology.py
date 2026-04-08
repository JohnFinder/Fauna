from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from src.tags.ontology import (
    ONTOLOGY_VERSION,
    TAG_REGISTRY,
    EvidenceLevel,
    EvidenceSource,
    TagEvent,
    TagLayer,
    TagMetadata,
    TagPrediction,
    get_tags_by_evidence,
    get_tags_by_layer,
)

_SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

_CAMELCASE_TAG = re.compile(r"^[a-z][a-zA-Z0-9]*$")

_EXPECTED_ACOUSTIC = frozenset(
    {"individualSignaturePresent", "highModulation", "narrowbandFm"}
)

_EXPECTED_FUNCTIONAL = frozenset(
    {
        "contactCall",
        "socialContactMaintenance",
        "flockLocalization",
        "sharedCallMatch",
        "unknownFunction",
        "socialApproachLikely",
        "highArousalLikely",
    }
)

_EXPECTED_CONTEXT = frozenset(
    {
        "isolationOrSeparationContext",
        "pairBondContextLikely",
        "visiblePartner",
        "audioOnlyPartner",
        "separation",
        "groupFlight",
        "perchedSocial",
        "operantTask",
        "playbackTrial",
    }
)


def test_ontology_version_is_semver() -> None:
    assert _SEMVER_PATTERN.match(ONTOLOGY_VERSION) is not None


def test_tag_registry_has_19_entries() -> None:
    assert len(TAG_REGISTRY) == 19


def test_all_tags_have_valid_layer() -> None:
    for meta in TAG_REGISTRY.values():
        assert isinstance(meta.layer, TagLayer)


def test_all_tags_have_valid_evidence() -> None:
    for meta in TAG_REGISTRY.values():
        assert isinstance(meta.evidence_level, EvidenceLevel)


def test_all_tags_have_nonempty_description() -> None:
    for meta in TAG_REGISTRY.values():
        assert meta.description.strip() != ""


def test_tag_names_are_camelcase() -> None:
    for name in TAG_REGISTRY:
        assert _CAMELCASE_TAG.match(name), name


def test_acoustic_tags_correct() -> None:
    assert set(get_tags_by_layer(TagLayer.ACOUSTIC)) == _EXPECTED_ACOUSTIC


def test_functional_tags_correct() -> None:
    assert set(get_tags_by_layer(TagLayer.FUNCTIONAL)) == _EXPECTED_FUNCTIONAL


def test_context_tags_correct() -> None:
    assert set(get_tags_by_layer(TagLayer.CONTEXT)) == _EXPECTED_CONTEXT


def test_supported_tags() -> None:
    supported = get_tags_by_evidence(EvidenceLevel.SUPPORTED)
    assert len(supported) == 16


def test_inferred_tags() -> None:
    inferred = get_tags_by_evidence(EvidenceLevel.INFERRED)
    assert len(inferred) == 3
    assert set(inferred) == {
        "socialApproachLikely",
        "highArousalLikely",
        "pairBondContextLikely",
    }


def test_no_do_not_claim_tags_in_v1() -> None:
    assert get_tags_by_evidence(EvidenceLevel.DO_NOT_CLAIM) == []


def test_tag_metadata_is_frozen() -> None:
    meta = next(iter(TAG_REGISTRY.values()))
    with pytest.raises(FrozenInstanceError):
        meta.name = "x"  # type: ignore[misc]


def test_tag_prediction_is_frozen() -> None:
    pred = TagPrediction(
        tag_name="contactCall",
        probability=0.5,
        evidence_source=EvidenceSource.LITERATURE,
    )
    with pytest.raises(FrozenInstanceError):
        pred.tag_name = "x"  # type: ignore[misc]


def test_tag_event_mutable() -> None:
    event = TagEvent(predictions=[], novelty_flag=False)
    event.novelty_flag = True
    assert event.novelty_flag is True


def test_tag_prediction_evidence_source_values() -> None:
    assert {e.value for e in EvidenceSource} == {
        "literature",
        "humanAnnotation",
        "modelInference",
        "playbackValidated",
    }


def test_layers_partition_all_tags() -> None:
    acoustic = set(get_tags_by_layer(TagLayer.ACOUSTIC))
    functional = set(get_tags_by_layer(TagLayer.FUNCTIONAL))
    context = set(get_tags_by_layer(TagLayer.CONTEXT))
    all_names = set(TAG_REGISTRY)
    assert acoustic | functional | context == all_names
    assert acoustic.isdisjoint(functional)
    assert acoustic.isdisjoint(context)
    assert functional.isdisjoint(context)


@pytest.mark.parametrize("layer", list(TagLayer))
def test_get_tags_by_layer_returns_sorted(layer: TagLayer) -> None:
    names = get_tags_by_layer(layer)
    assert names == sorted(names)
