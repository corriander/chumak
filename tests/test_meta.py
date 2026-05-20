"""Meta builder: provenance pass-through, hashing, AIMessage extraction."""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage

from chumak.handlers.base import HandlerResult
from chumak.handlers.types import HandlerType
from chumak.meta import build_meta
from chumak.profile import Profile
from chumak.response import ArtefactRef, Provenance


def _profile() -> Profile:
    return Profile(
        name="claude",
        handler=HandlerType.LANGCHAIN,
        model="anthropic:claude-opus-4-7",
        prompt_version="mission-title@v1",
    )


def test_no_provenance_yields_empty_artefact_fields() -> None:
    handler_result = HandlerResult(payload=None, raw=None, rendered_prompt="hello")
    meta = build_meta(profile=_profile(), handler_result=handler_result)
    assert meta.artefact_type is None
    assert meta.artefact_id is None
    assert meta.derived_from == []
    assert meta.produced_by.profile == "claude"
    assert meta.produced_by.prompt_version == "mission-title@v1"


def test_provenance_populates_artefact_fields() -> None:
    handler_result = HandlerResult(payload=None, raw=None, rendered_prompt="hello")
    provenance = Provenance(
        artefact_type="mission_title@v1",
        artefact_id="screenshot:2026-05-20T12:34:56Z",
        derived_from=[ArtefactRef(artefact_type="screenshot@v1", artefact_id="abc123")],
        prompt_template_sha256="deadbeef",
    )
    meta = build_meta(profile=_profile(), handler_result=handler_result, provenance=provenance)
    assert meta.artefact_type == "mission_title@v1"
    assert meta.artefact_id == "screenshot:2026-05-20T12:34:56Z"
    assert meta.derived_from[0].artefact_type == "screenshot@v1"
    assert meta.produced_by.prompt_template_sha256 == "deadbeef"


def test_prompt_actual_sha256_hashes_rendered_prompt() -> None:
    prompt = "hello world"
    expected = hashlib.sha256(prompt.encode()).hexdigest()
    handler_result = HandlerResult(payload=None, raw=None, rendered_prompt=prompt)
    meta = build_meta(profile=_profile(), handler_result=handler_result)
    assert meta.produced_by.prompt_actual_sha256 == expected


def test_cost_extracted_from_aimessage_usage_metadata() -> None:
    raw = AIMessage(
        content="...",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    handler_result = HandlerResult(payload=None, raw=raw, rendered_prompt="hi")
    meta = build_meta(profile=_profile(), handler_result=handler_result)
    assert meta.cost.tokens_in == 100
    assert meta.cost.tokens_out == 50


def test_cost_empty_when_raw_is_not_aimessage() -> None:
    handler_result = HandlerResult(payload=None, raw={"not": "an aimessage"}, rendered_prompt="hi")
    meta = build_meta(profile=_profile(), handler_result=handler_result)
    assert meta.cost.tokens_in is None
    assert meta.cost.tokens_out is None
