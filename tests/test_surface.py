"""`infer()` surface: handler dispatch + meta wiring.

Uses the `stub_handler` fixture from conftest to swap the LangChain handler
for a deterministic fake, so the surface tests stay hermetic.
"""

from __future__ import annotations

from chumak.handlers.base import HandlerResult
from chumak.handlers.types import HandlerType
from chumak.profile import Profile
from chumak.response import Provenance
from chumak.surface import infer
from tests.conftest import SampleOutput


def _profile() -> Profile:
    return Profile(
        name="claude",
        handler=HandlerType.LANGCHAIN,
        model="anthropic:claude-opus-4-7",
    )


def test_infer_returns_payload_and_meta(stub_handler) -> None:
    payload = SampleOutput(title="Haul ore", bounty=42_000)
    stub_handler(HandlerResult(payload=payload, raw=None, rendered_prompt="extract"))

    result = infer(
        prompt="extract",
        output_schema=SampleOutput,
        profile=_profile(),
    )

    assert result.payload is payload
    assert result.meta.produced_by.profile == "claude"
    assert result.meta.produced_by.model == "anthropic:claude-opus-4-7"
    assert result.meta.artefact_type is None  # no provenance supplied
    assert result.meta.generated_at is not None


def test_infer_without_schema_returns_text_payload(stub_handler) -> None:
    # Untyped call: output_schema omitted, payload is the model's plain text.
    stub_handler(HandlerResult(payload="pong", raw=None, rendered_prompt="ping"))

    result = infer(prompt="ping", profile=_profile())

    assert result.payload == "pong"
    assert result.meta.produced_by.profile == "claude"
    assert result.meta.generated_at is not None


def test_infer_with_provenance_populates_artefact_fields(stub_handler) -> None:
    payload = SampleOutput(title="Haul ore", bounty=42_000)
    stub_handler(HandlerResult(payload=payload, raw=None, rendered_prompt="extract"))

    result = infer(
        prompt="extract",
        output_schema=SampleOutput,
        profile=_profile(),
        provenance=Provenance(
            artefact_type="mission_title@v1",
            artefact_id="screenshot:abc",
        ),
    )

    assert result.meta.artefact_type == "mission_title@v1"
    assert result.meta.artefact_id == "screenshot:abc"
