"""LangChainHandler: typed vs untyped dispatch, hermetic.

`init_chat_model` is monkeypatched to a fake chat model so both branches of
the handler are exercised without a live backend (the live wiring is covered
separately in `test_langchain_live.py`, behind `--integration`).

Also covers `resolve_model`, the public model-construction path the handler
shares with consumers that orchestrate their own loop.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from chumak.attachments import Attachment
from chumak.errors import ProfileCapabilityError
from chumak.handlers.langchain import LangChainHandler, resolve_model
from chumak.handlers.types import HandlerType, PromptDelivery
from chumak.profile import Profile
from tests.conftest import solid_png


class ColourTag(BaseModel):
    colour: str
    is_warm: bool


class _FakeStructured:
    """What `.with_structured_output(...)` returns: an invokable that yields the
    include_raw envelope `{parsed, raw, parsing_error}`."""

    def __init__(self, parsed: object, raw: object, parsing_error: object = None) -> None:
        self._envelope = {"parsed": parsed, "raw": raw, "parsing_error": parsing_error}
        self.last_input: Any = None

    def invoke(self, model_input: Any) -> dict:
        self.last_input = model_input
        return self._envelope


class _FakeModel:
    """Stands in for the object `init_chat_model` returns."""

    def __init__(self, *, text: str, structured: _FakeStructured | None = None) -> None:
        self._message = AIMessage(content=text)
        self._structured = structured
        self.structured_schema: type[BaseModel] | None = None
        self.last_input: Any = None

    def invoke(self, model_input: Any) -> AIMessage:
        self.last_input = model_input
        return self._message

    def with_structured_output(self, schema: type[BaseModel], *, include_raw: bool):
        self.structured_schema = schema
        assert include_raw is True
        return self._structured


def _profile(**overrides: object) -> Profile:
    fields: dict[str, object] = {
        "name": "local-openai",
        "handler": HandlerType.LANGCHAIN,
        "model": "openai:mistral-7b-instruct-v0.3",
        "temperature": 0.0,
        "api_key": "sk-local-unused",
        "model_kwargs": {"base_url": "http://edge/v1"},
    }
    fields.update(overrides)
    return Profile.model_validate(fields)


def test_api_key_reaches_init_chat_model_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_init(model: str, **kwargs: object) -> _FakeModel:
        seen.update(kwargs)
        return _FakeModel(text="ok")

    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", fake_init)
    LangChainHandler().execute(prompt="ping", output_schema=None, profile=_profile())

    assert seen["api_key"] == "sk-local-unused"
    assert seen["base_url"] == "http://edge/v1"


def test_unset_api_key_is_left_to_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `api_key` kwarg at all, so the provider class reads its own env var."""
    seen: dict[str, object] = {}

    def fake_init(model: str, **kwargs: object) -> _FakeModel:
        seen.update(kwargs)
        return _FakeModel(text="ok")

    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", fake_init)
    LangChainHandler().execute(prompt="ping", output_schema=None, profile=_profile(api_key=None))

    assert "api_key" not in seen


def test_untyped_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(text="hello from the edge")
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    result = LangChainHandler().execute(prompt="ping", output_schema=None, profile=_profile())

    assert result.payload == "hello from the edge"
    assert isinstance(result.raw, AIMessage)
    assert result.rendered_prompt == "ping"
    assert result.attachments == []
    # Text-only calls keep the bare-string input — unchanged pre-attachments behaviour.
    assert model.last_input == "ping"
    # No structured-output translation was applied on the untyped path.
    assert model.structured_schema is None


def test_typed_returns_validated_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = ColourTag(colour="crimson", is_warm=True)
    structured = _FakeStructured(parsed=parsed, raw=AIMessage(content="{...}"))
    model = _FakeModel(text="unused", structured=structured)
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    result = LangChainHandler().execute(
        prompt="classify", output_schema=ColourTag, profile=_profile()
    )

    assert result.payload is parsed
    assert model.structured_schema is ColourTag


def test_typed_raises_on_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    structured = _FakeStructured(parsed=None, raw=AIMessage(content="oops"), parsing_error="boom")
    model = _FakeModel(text="unused", structured=structured)
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    with pytest.raises(ValueError, match="Structured output parsing failed"):
        LangChainHandler().execute(prompt="x", output_schema=ColourTag, profile=_profile())


# --- resolve_model ----------------------------------------------------------


def _capturing_init(model: object) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """A stand-in for `init_chat_model` that records every (identifier, kwargs)."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def _init(identifier: str, **kwargs: Any) -> object:
        calls.append((identifier, kwargs))
        return model

    return _init, calls


def test_resolve_model_assembles_kwargs_with_model_kwargs_winning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(text="unused")
    init, calls = _capturing_init(model)
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", init)

    profile = Profile(
        name="p",
        handler=HandlerType.LANGCHAIN,
        model="openai:gpt-5",
        temperature=0.7,
        max_tokens=256,
        model_kwargs={"temperature": 0.0, "base_url": "http://edge/v1"},
    )

    assert resolve_model(profile) is model
    assert calls == [
        ("openai:gpt-5", {"temperature": 0.0, "max_tokens": 256, "base_url": "http://edge/v1"})
    ]


def test_resolve_model_and_execute_share_one_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model `resolve_model` hands out is configured identically to the one
    `execute` builds for the same profile — one code path, observed via the
    `init_chat_model` call each makes.
    """
    model = _FakeModel(text="pong")
    init, calls = _capturing_init(model)
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", init)
    profile = _profile()

    resolve_model(profile)
    LangChainHandler().execute(prompt="ping", output_schema=None, profile=profile)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    # Parity alone would pass if both dropped the key; pin that it's there.
    assert calls[0][1]["api_key"] == "sk-local-unused"


def test_resolve_model_rejects_subprocess_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    init, calls = _capturing_init(object())
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", init)
    profile = Profile(
        name="cli",
        handler=HandlerType.SUBPROCESS,
        model="claude-opus-4-7",
        command="claude --print",
        prompt_delivery=PromptDelivery.STDIN,
    )

    with pytest.raises(ProfileCapabilityError, match="only 'langchain' profiles"):
        resolve_model(profile)
    assert calls == []  # never reached init_chat_model


# --- attachments ------------------------------------------------------------


def _expected_image_part(path: Path) -> dict[str, Any]:
    return {
        "type": "image",
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mime_type": "image/png",
    }


def test_untyped_with_attachment_sends_multimodal_message(
    monkeypatch: pytest.MonkeyPatch, red_png: Path
) -> None:
    model = _FakeModel(text="a red square")
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    result = LangChainHandler().execute(
        prompt="What colour is this?",
        output_schema=None,
        profile=_profile(),
        attachments=[Attachment(path=red_png)],
    )

    # One HumanMessage of standard content blocks: text first, then the image.
    assert isinstance(model.last_input, list) and len(model.last_input) == 1
    message = model.last_input[0]
    assert isinstance(message, HumanMessage)
    assert message.content == [
        {"type": "text", "text": "What colour is this?"},
        _expected_image_part(red_png),
    ]
    # Provenance: text hash unchanged in meaning, image recorded by digest.
    assert result.rendered_prompt == "What colour is this?"
    assert [d.model_dump() for d in result.attachments] == [
        {"sha256": hashlib.sha256(red_png.read_bytes()).hexdigest(), "mime": "image/png"}
    ]


def test_typed_with_attachment_routes_through_structured_output(
    monkeypatch: pytest.MonkeyPatch, red_png: Path
) -> None:
    parsed = ColourTag(colour="red", is_warm=True)
    structured = _FakeStructured(parsed=parsed, raw=AIMessage(content="{...}"))
    model = _FakeModel(text="unused", structured=structured)
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    result = LangChainHandler().execute(
        prompt="classify",
        output_schema=ColourTag,
        profile=_profile(),
        attachments=[Attachment(path=red_png)],
    )

    assert result.payload is parsed
    assert model.structured_schema is ColourTag
    assert isinstance(structured.last_input, list)
    assert structured.last_input[0].content[1] == _expected_image_part(red_png)
    assert len(result.attachments) == 1


def test_multiple_attachments_keep_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    red.write_bytes(solid_png(4, 4, (255, 0, 0)))
    blue.write_bytes(solid_png(4, 4, (0, 0, 255)))
    model = _FakeModel(text="two squares")
    monkeypatch.setattr("chumak.handlers.langchain.init_chat_model", lambda m, **k: model)

    result = LangChainHandler().execute(
        prompt="compare",
        output_schema=None,
        profile=_profile(),
        attachments=[Attachment(path=red), Attachment(path=blue)],
    )

    parts = model.last_input[0].content
    assert [part["type"] for part in parts] == ["text", "image", "image"]
    assert [d.sha256 for d in result.attachments] == [
        hashlib.sha256(red.read_bytes()).hexdigest(),
        hashlib.sha256(blue.read_bytes()).hexdigest(),
    ]
