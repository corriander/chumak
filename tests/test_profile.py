"""Profile validation rules: handler-discriminated mutual exclusion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chumak.handlers.types import HandlerType, PromptDelivery
from chumak.profile import Profile


def test_langchain_profile_minimal_is_valid() -> None:
    p = Profile(name="claude", handler=HandlerType.LANGCHAIN, model="anthropic:claude-opus-4-7")
    assert p.is_subprocess is False
    assert p.model_kwargs == {}


def test_subprocess_profile_requires_command_and_delivery() -> None:
    with pytest.raises(ValidationError, match="must set `command`"):
        Profile(name="cli", handler=HandlerType.SUBPROCESS, model="claude-opus-4-7")


def test_subprocess_profile_rejects_model_kwargs() -> None:
    with pytest.raises(ValidationError, match="cannot set `model_kwargs`"):
        Profile(
            name="cli",
            handler=HandlerType.SUBPROCESS,
            model="claude-opus-4-7",
            command="claude --print",
            prompt_delivery=PromptDelivery.STDIN,
            model_kwargs={"top_p": 0.9},
        )


def test_subprocess_profile_rejects_api_key() -> None:
    with pytest.raises(ValidationError, match="cannot set `api_key`"):
        Profile(
            name="cli",
            handler=HandlerType.SUBPROCESS,
            model="claude-opus-4-7",
            command="claude --print",
            prompt_delivery=PromptDelivery.STDIN,
            api_key="sk-secret-value",
        )


def test_api_key_is_masked_wherever_the_profile_is_rendered() -> None:
    p = Profile(
        name="claude",
        handler=HandlerType.LANGCHAIN,
        model="anthropic:claude-opus-4-7",
        api_key="sk-secret-value",
    )
    for rendered in (repr(p), str(p), p.model_dump_json(), str(p.model_dump())):
        assert "sk-secret-value" not in rendered
    assert p.api_key is not None
    assert p.api_key.get_secret_value() == "sk-secret-value"


def test_model_kwargs_api_key_is_rejected_with_the_new_location() -> None:
    """The clean break: the old location fails loudly rather than being
    silently forwarded (and rendered in the clear)."""
    with pytest.raises(ValidationError, match="top-level `api_key`") as excinfo:
        Profile(
            name="claude",
            handler=HandlerType.LANGCHAIN,
            model="anthropic:claude-opus-4-7",
            model_kwargs={"api_key": "sk-secret-value"},
        )
    assert "sk-secret-value" not in str(excinfo.value)


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": "sk-secret-value"},  # a key routed to the wrong field
        {"apikey": "sk-secret-value"},  # a misspelt field (extra="forbid")
    ],
)
def test_validation_errors_do_not_echo_input_values(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({"name": "claude", "handler": "langchain", "model": "m", **fields})
    assert "sk-secret-value" not in str(excinfo.value)


def test_langchain_profile_rejects_subprocess_only_fields() -> None:
    with pytest.raises(ValidationError, match="only valid when"):
        Profile(
            name="claude",
            handler=HandlerType.LANGCHAIN,
            model="anthropic:claude-opus-4-7",
            command="claude --print",
        )


def test_subprocess_profile_full_shape() -> None:
    p = Profile(
        name="cli",
        handler=HandlerType.SUBPROCESS,
        model="claude-opus-4-7",
        command="claude --print --model claude-opus-4-7",
        prompt_delivery=PromptDelivery.STDIN,
        timeout=120.0,
    )
    assert p.is_subprocess is True
    assert p.timeout == 120.0


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Profile.model_validate(
            {
                "name": "claude",
                "handler": "langchain",
                "model": "anthropic:claude-opus-4-7",
                "nonsense": "extra",
            }
        )
