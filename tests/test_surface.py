"""`infer()` surface: handler dispatch + meta wiring.

Uses the `stub_handler` fixture from conftest to swap the LangChain handler
for a deterministic fake, so the surface tests stay hermetic.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from chumak.attachments import Attachment, AttachmentDigest
from chumak.handlers import HANDLER_REGISTRY
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


def test_infer_records_no_prompt_hash_when_handler_omits_rendered_prompt(stub_handler) -> None:
    # A handler may augment the prompt without reporting what it sent; the
    # caller's text is then not known to be what reached the transport.
    stub_handler(HandlerResult(payload="ok"))

    result = infer(prompt="ping", profile=_profile())

    assert result.meta.produced_by.prompt_actual_sha256 is None


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


def test_infer_stamps_handler_attachment_digests_into_meta(stub_handler) -> None:
    digest = AttachmentDigest(sha256="ab" * 32, mime="image/png")
    stub_handler(
        HandlerResult(
            payload="a red square", raw=None, rendered_prompt="what", attachments=[digest]
        )
    )

    result = infer(prompt="what", profile=_profile())

    assert result.meta.produced_by.attachments == [digest]


class _TextOnlyHandler:
    """A handler written before attachments existed: no `attachments` keyword."""

    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult:
        return HandlerResult(payload="pong", rendered_prompt=prompt)


@pytest.fixture
def text_only_handler() -> Iterator[None]:
    original = HANDLER_REGISTRY[HandlerType.LANGCHAIN]
    # Deliberately off-protocol: the point is that infer() still accepts it.
    HANDLER_REGISTRY[HandlerType.LANGCHAIN] = _TextOnlyHandler  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        HANDLER_REGISTRY[HandlerType.LANGCHAIN] = original


def test_text_only_call_still_dispatches_to_a_pre_attachment_handler(text_only_handler) -> None:
    result = infer(prompt="ping", profile=_profile())

    assert result.payload == "pong"
    assert result.meta.produced_by.attachments == []


def test_attachments_to_a_pre_attachment_handler_fail_loudly(text_only_handler, red_png) -> None:
    with pytest.raises(TypeError, match="attachments"):
        infer(prompt="what", attachments=[Attachment(path=red_png)], profile=_profile())
