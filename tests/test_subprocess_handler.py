"""SubprocessHandler: fenced-JSON parsing, schema injection, error paths.

The handler shells out via `subprocess.run`. We use `pytest-mock`'s
`mocker.patch.object` to inject a fake `CompletedProcess` so tests stay
hermetic and fast.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
from pydantic import BaseModel

from chumak.handlers.subprocess import SubprocessHandler, _build_subprocess_prompt
from chumak.handlers.types import HandlerType, PromptDelivery
from chumak.profile import Profile


class Out(BaseModel):
    name: str
    count: int


def _profile(delivery: PromptDelivery = PromptDelivery.STDIN) -> Profile:
    return Profile(
        name="cli",
        handler=HandlerType.SUBPROCESS,
        model="claude-opus-4-7",
        command="claude --print",
        prompt_delivery=delivery,
        timeout=30.0,
    )


def _fake_completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude", "--print"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_build_subprocess_prompt_embeds_schema() -> None:
    rendered = _build_subprocess_prompt("hello", Out)
    assert "hello" in rendered
    assert "JSON Schema" in rendered
    assert '"name"' in rendered  # field from schema landed in the prompt


def test_execute_without_schema_raises(mocker) -> None:
    # Untyped inference is a langchain-handler capability; a subprocess profile
    # has no schema to inject or validate against, so it must refuse loudly
    # (and never shell out).
    handler = SubprocessHandler()
    run = mocker.patch.object(handler, "_run")
    with pytest.raises(ValueError, match="requires an output_schema"):
        handler.execute(prompt="extract", output_schema=None, profile=_profile())
    run.assert_not_called()


def test_execute_parses_plain_json(mocker) -> None:
    handler = SubprocessHandler()
    mocker.patch.object(
        handler,
        "_run",
        return_value=_fake_completed('{"name": "ore", "count": 7}'),
    )
    result = handler.execute(prompt="extract", output_schema=Out, profile=_profile())
    assert isinstance(result.payload, Out)
    assert result.payload.name == "ore"
    assert result.payload.count == 7


def test_execute_parses_fenced_json(mocker) -> None:
    handler = SubprocessHandler()
    mocker.patch.object(
        handler,
        "_run",
        return_value=_fake_completed('```json\n{"name": "ore", "count": 7}\n```'),
    )
    result = handler.execute(prompt="extract", output_schema=Out, profile=_profile())
    assert result.payload.count == 7


def test_execute_raises_on_non_zero_exit(mocker) -> None:
    handler = SubprocessHandler()
    mocker.patch.object(
        handler, "_run", return_value=_fake_completed("", returncode=2, stderr="boom")
    )
    with pytest.raises(RuntimeError, match="exited 2"):
        handler.execute(prompt="extract", output_schema=Out, profile=_profile())


def test_execute_raises_on_invalid_json(mocker) -> None:
    handler = SubprocessHandler()
    mocker.patch.object(handler, "_run", return_value=_fake_completed("not json at all"))
    with pytest.raises(ValueError, match="non-JSON output"):
        handler.execute(prompt="extract", output_schema=Out, profile=_profile())


def test_execute_raises_on_schema_mismatch(mocker) -> None:
    handler = SubprocessHandler()
    mocker.patch.object(
        handler,
        "_run",
        return_value=_fake_completed('{"name": "ore"}'),  # missing `count`
    )
    with pytest.raises(ValueError, match="schema validation"):
        handler.execute(prompt="extract", output_schema=Out, profile=_profile())


def test_execute_rejects_non_subprocess_profile() -> None:
    bad_profile = Profile(
        name="not-cli",
        handler=HandlerType.LANGCHAIN,
        model="anthropic:claude-opus-4-7",
    )
    with pytest.raises(ValueError, match="non-subprocess"):
        SubprocessHandler().execute(prompt="x", output_schema=Out, profile=bad_profile)


def test_arg_delivery_appends_prompt_to_argv(mocker: Any) -> None:
    """ARG delivery: the prompt is passed as the last positional argv."""
    handler = SubprocessHandler()
    seen: dict[str, Any] = {}

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        seen["argv"] = argv
        return _fake_completed('{"name": "x", "count": 1}')

    mocker.patch("chumak.handlers.subprocess.subprocess.run", side_effect=fake_run)
    handler.execute(
        prompt="extract",
        output_schema=Out,
        profile=_profile(delivery=PromptDelivery.ARG),
    )
    assert seen["argv"][0] == "claude"
    assert "extract" in seen["argv"][-1]
