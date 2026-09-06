"""Meta builder: provenance pass-through, hashing, AIMessage extraction.

`build_meta` is public — the last test pins the contract that a consumer
stamping its own call gets the same envelope `infer()` would have produced.
"""

from __future__ import annotations

import hashlib

from langchain_core.messages import AIMessage

from chumak.handlers.base import HandlerResult
from chumak.handlers.types import HandlerType
from chumak.meta import build_meta
from chumak.profile import Profile
from chumak.response import ArtefactRef, Provenance
from chumak.surface import infer


def _profile() -> Profile:
    return Profile(
        name="claude",
        handler=HandlerType.LANGCHAIN,
        model="anthropic:claude-opus-4-7",
        prompt_version="mission-title@v1",
    )


def test_no_provenance_yields_empty_artefact_fields() -> None:
    meta = build_meta(None, profile=_profile(), prompt="hello")
    assert meta.artefact_type is None
    assert meta.artefact_id is None
    assert meta.derived_from == []
    assert meta.produced_by.profile == "claude"
    assert meta.produced_by.prompt_version == "mission-title@v1"


def test_provenance_populates_artefact_fields() -> None:
    provenance = Provenance(
        artefact_type="mission_title@v1",
        artefact_id="screenshot:2026-05-20T12:34:56Z",
        derived_from=[ArtefactRef(artefact_type="screenshot@v1", artefact_id="abc123")],
        prompt_template_sha256="deadbeef",
    )
    meta = build_meta(None, profile=_profile(), prompt="hello", provenance=provenance)
    assert meta.artefact_type == "mission_title@v1"
    assert meta.artefact_id == "screenshot:2026-05-20T12:34:56Z"
    assert meta.derived_from[0].artefact_type == "screenshot@v1"
    assert meta.produced_by.prompt_template_sha256 == "deadbeef"


def test_prompt_actual_sha256_hashes_rendered_prompt() -> None:
    prompt = "hello world"
    expected = hashlib.sha256(prompt.encode()).hexdigest()
    meta = build_meta(None, profile=_profile(), prompt=prompt)
    assert meta.produced_by.prompt_actual_sha256 == expected


def test_provenance_carries_neither_model_kwargs_nor_the_key() -> None:
    """`meta` is what consumers persist, so nothing credential-shaped rides on it."""
    profile = Profile(
        name="local",
        handler=HandlerType.LANGCHAIN,
        model="openai:qwen",
        api_key="sk-secret-value",
        model_kwargs={"default_headers": {"Authorization": "Bearer sk-header-secret"}},
    )
    meta = build_meta(None, profile=profile, prompt="hi")

    assert "model_kwargs" not in type(meta.produced_by).model_fields
    dumped = meta.model_dump_json()
    assert "sk-secret-value" not in dumped
    assert "sk-header-secret" not in dumped


def test_cost_extracted_from_aimessage_usage_metadata() -> None:
    raw = AIMessage(
        content="...",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    meta = build_meta(raw, profile=_profile(), prompt="hi")
    assert meta.cost.tokens_in == 100
    assert meta.cost.tokens_out == 50


def test_cost_empty_when_raw_is_not_aimessage() -> None:
    meta = build_meta({"not": "an aimessage"}, profile=_profile(), prompt="hi")
    assert meta.cost.tokens_in is None
    assert meta.cost.tokens_out is None


def test_consumer_stamped_meta_matches_infer(stub_handler) -> None:
    """Round-trip: a consumer that made the call itself and stamps it with
    `build_meta` gets the envelope `infer()` produces for the same
    profile/prompt/response — `generated_at` aside, which is wall-clock.
    """
    profile = _profile()
    prompt = "extract the title"
    raw = AIMessage(
        content="Haul ore",
        usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    )
    provenance = Provenance(artefact_type="mission_title@v1", artefact_id="screenshot:abc")

    stub_handler(HandlerResult(payload="Haul ore", raw=raw, rendered_prompt=prompt))
    via_infer = infer(prompt=prompt, profile=profile, provenance=provenance).meta

    via_consumer = build_meta(raw, profile=profile, prompt=prompt, provenance=provenance)

    exclude = {"generated_at"}
    assert via_consumer.model_dump(exclude=exclude) == via_infer.model_dump(exclude=exclude)
