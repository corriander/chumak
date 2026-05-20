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
            model_kwargs={"api_key": "sk-..."},
        )


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
