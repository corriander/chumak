"""LangChainHandler: typed vs untyped dispatch, hermetic.

`init_chat_model` is monkeypatched to a fake chat model so both branches of
the handler are exercised without a live backend (the live wiring is covered
separately in `test_langchain_live.py`, behind `--integration`).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from chumak.handlers.langchain import LangChainHandler
from chumak.handlers.types import HandlerType
from chumak.profile import Profile


class ColourTag(BaseModel):
    colour: str
    is_warm: bool


class _FakeStructured:
    """What `.with_structured_output(...)` returns: an invokable that yields the
    include_raw envelope `{parsed, raw, parsing_error}`."""

    def __init__(self, parsed: object, raw: object, parsing_error: object = None) -> None:
        self._envelope = {"parsed": parsed, "raw": raw, "parsing_error": parsing_error}

    def invoke(self, prompt: str) -> dict:
        return self._envelope


class _FakeModel:
    """Stands in for the object `init_chat_model` returns."""

    def __init__(self, *, text: str, structured: _FakeStructured | None = None) -> None:
        self._message = AIMessage(content=text)
        self._structured = structured
        self.structured_schema: type[BaseModel] | None = None

    def invoke(self, prompt: str) -> AIMessage:
        return self._message

    def with_structured_output(self, schema: type[BaseModel], *, include_raw: bool):
        self.structured_schema = schema
        assert include_raw is True
        return self._structured


def _profile() -> Profile:
    return Profile(
        name="local-openai",
        handler=HandlerType.LANGCHAIN,
        model="openai:mistral-7b-instruct-v0.3",
        temperature=0.0,
        model_kwargs={"base_url": "http://edge/v1", "api_key": "sk-local-unused"},
    )


def test_untyped_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(text="hello from the edge")
    monkeypatch.setattr(
        "chumak.handlers.langchain.init_chat_model", lambda m, **k: model
    )

    result = LangChainHandler().execute(prompt="ping", output_schema=None, profile=_profile())

    assert result.payload == "hello from the edge"
    assert isinstance(result.raw, AIMessage)
    assert result.rendered_prompt == "ping"
    # No structured-output translation was applied on the untyped path.
    assert model.structured_schema is None


def test_typed_returns_validated_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = ColourTag(colour="crimson", is_warm=True)
    structured = _FakeStructured(parsed=parsed, raw=AIMessage(content="{...}"))
    model = _FakeModel(text="unused", structured=structured)
    monkeypatch.setattr(
        "chumak.handlers.langchain.init_chat_model", lambda m, **k: model
    )

    result = LangChainHandler().execute(
        prompt="classify", output_schema=ColourTag, profile=_profile()
    )

    assert result.payload is parsed
    assert model.structured_schema is ColourTag


def test_typed_raises_on_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    structured = _FakeStructured(parsed=None, raw=AIMessage(content="oops"), parsing_error="boom")
    model = _FakeModel(text="unused", structured=structured)
    monkeypatch.setattr(
        "chumak.handlers.langchain.init_chat_model", lambda m, **k: model
    )

    with pytest.raises(ValueError, match="Structured output parsing failed"):
        LangChainHandler().execute(prompt="x", output_schema=ColourTag, profile=_profile())
