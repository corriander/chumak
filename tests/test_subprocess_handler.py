"""SubprocessHandler: fenced-JSON parsing, schema injection, error paths.

The handler shells out via `subprocess.run`. We use `pytest-mock`'s
`mocker.patch.object` to inject a fake `CompletedProcess` so tests stay
hermetic and fast.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from chumak.attachments import Attachment
from chumak.errors import ProfileCapabilityError
from chumak.handlers.subprocess import SubprocessHandler, _build_subprocess_prompt
from chumak.handlers.types import HandlerType, PromptDelivery
from chumak.profile import Profile
from chumak.surface import infer
from tests.conftest import solid_png


class Out(BaseModel):
    name: str
    count: int


def _profile(
    delivery: PromptDelivery = PromptDelivery.STDIN, attachment_ref: str | None = None
) -> Profile:
    return Profile(
        name="cli",
        handler=HandlerType.SUBPROCESS,
        model="claude-opus-4-7",
        command="claude --print",
        prompt_delivery=delivery,
        timeout=30.0,
        attachment_ref=attachment_ref,
    )


def _fake_completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude", "--print"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _completed_bytes(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    """What `subprocess.run` itself returns: the pipes are bytes."""
    return subprocess.CompletedProcess(
        args=argv, returncode=0, stdout=b'{"name": "x", "count": 1}', stderr=b""
    )


def test_non_utf8_output_is_a_clear_error(mocker) -> None:
    mocker.patch(
        "chumak.handlers.subprocess.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["claude"], returncode=0, stdout=b'{"name": "caf\xe9", "count": 1}', stderr=b""
        ),
    )
    with pytest.raises(ValueError, match="non-UTF-8 output"):
        SubprocessHandler().execute(prompt="extract", output_schema=Out, profile=_profile())


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


def test_execute_rejects_attachments_without_attachment_ref(mocker, red_png) -> None:
    # Without a template there's no way to reference the file, so a non-empty
    # `attachments` must refuse loudly (and never shell out) rather than be
    # silently dropped.
    handler = SubprocessHandler()
    run = mocker.patch.object(handler, "_run")
    with pytest.raises(ProfileCapabilityError, match="set `attachment_ref`"):
        handler.execute(
            prompt="extract",
            output_schema=Out,
            profile=_profile(),
            attachments=[Attachment(path=red_png)],
        )
    run.assert_not_called()


def _swatch(directory: Path, name: str = "swatch.png") -> Attachment:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(solid_png(4, 4, (0, 0, 255)))
    return Attachment(path=path)


def test_references_follow_the_prompt_in_order_before_the_schema(mocker, tmp_path) -> None:
    first, second = _swatch(tmp_path / "a"), _swatch(tmp_path / "b")
    handler = SubprocessHandler()
    run = mocker.patch.object(
        handler, "_run", return_value=_fake_completed('{"name": "x", "count": 1}')
    )

    result = handler.execute(
        prompt="extract",
        output_schema=Out,
        profile=_profile(attachment_ref="@{path}"),
        attachments=[first, second],
    )

    sent = run.call_args.args[1]
    expected_head = f"extract\n\n@{first.path.absolute()}\n@{second.path.absolute()}\n\n"
    assert sent.startswith(expected_head)
    assert sent[len(expected_head) :] == _build_subprocess_prompt("", Out)[2:]
    assert result.rendered_prompt == sent
    assert result.attachments_referenced == [first.digest(), second.digest()]
    assert result.attachments == []  # nothing was sent by chumak itself


def test_attachment_ref_leaves_a_text_only_prompt_unchanged(mocker) -> None:
    handler = SubprocessHandler()
    run = mocker.patch.object(
        handler, "_run", return_value=_fake_completed('{"name": "x", "count": 1}')
    )

    result = handler.execute(
        prompt="extract", output_schema=Out, profile=_profile(attachment_ref="@{path}")
    )

    assert run.call_args.args[1] == _build_subprocess_prompt("extract", Out)
    assert result.attachments_referenced == []


def test_relative_attachment_path_is_referenced_absolutely(mocker, tmp_path, monkeypatch) -> None:
    # The CLI runs in the caller's working directory, which may change between
    # naming the attachment and the call.
    _swatch(tmp_path)
    monkeypatch.chdir(tmp_path)
    attachment = Attachment(path=Path("swatch.png"))
    handler = SubprocessHandler()
    run = mocker.patch.object(
        handler, "_run", return_value=_fake_completed('{"name": "x", "count": 1}')
    )

    handler.execute(
        prompt="extract",
        output_schema=Out,
        profile=_profile(attachment_ref="{path}"),
        attachments=[attachment],
    )

    assert f"\n\n{tmp_path / 'swatch.png'}\n\n" in run.call_args.args[1]


def test_awkward_paths_are_substituted_literally(mocker, tmp_path) -> None:
    # Spaces, quotes, `@`, braces and non-ASCII reach the CLI as they are:
    # quoting is the template's job, and braces in the path are not placeholders.
    attachment = _swatch(tmp_path / "a b 'q' @{path} ✓", name="shot {x}.png")
    handler = SubprocessHandler()
    run = mocker.patch.object(
        handler, "_run", return_value=_fake_completed('{"name": "x", "count": 1}')
    )

    handler.execute(
        prompt="extract",
        output_schema=Out,
        profile=_profile(attachment_ref='@"{path}"'),
        attachments=[attachment],
    )

    assert f'\n\n@"{attachment.path.absolute()}"\n\n' in run.call_args.args[1]


def test_path_with_a_line_break_is_rejected_before_spawn(mocker, tmp_path) -> None:
    # Not creatable on every OS, so built without validation.
    attachment = Attachment.model_construct(path=tmp_path / "a\nb.png", mime="image/png")
    handler = SubprocessHandler()
    run = mocker.patch.object(handler, "_run")

    with pytest.raises(ValueError, match="line break"):
        handler.execute(
            prompt="extract",
            output_schema=Out,
            profile=_profile(attachment_ref="{path}"),
            attachments=[attachment],
        )
    run.assert_not_called()


def test_unreadable_attachment_fails_before_spawn(mocker, tmp_path) -> None:
    # No digest means no provenance for the reference, so the CLI never runs.
    attachment = _swatch(tmp_path)
    attachment.path.unlink()
    handler = SubprocessHandler()
    run = mocker.patch.object(handler, "_run")

    with pytest.raises(FileNotFoundError):
        handler.execute(
            prompt="extract",
            output_schema=Out,
            profile=_profile(attachment_ref="{path}"),
            attachments=[attachment],
        )
    run.assert_not_called()


def test_infer_records_referenced_attachments_apart_from_sent_ones(mocker, tmp_path) -> None:
    attachment = _swatch(tmp_path)
    seen: dict[str, Any] = {}

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        seen["prompt"] = kw["input"].decode("utf-8")
        return _completed_bytes(argv)

    mocker.patch("chumak.handlers.subprocess.subprocess.run", side_effect=fake_run)

    produced_by = infer(
        prompt="extract",
        attachments=[attachment],
        output_schema=Out,
        profile=_profile(attachment_ref="{path}"),
    ).meta.produced_by

    assert produced_by.attachments == []
    assert produced_by.attachments_referenced == [attachment.digest()]
    # The hash covers the references, since they're part of the text sent.
    assert produced_by.prompt_actual_sha256 == hashlib.sha256(seen["prompt"].encode()).hexdigest()
    assert str(attachment.path.absolute()) in seen["prompt"]


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


def test_execute_parses_bare_json_with_fence_inside_string(mocker) -> None:
    # Regression: stdout is valid bare JSON whose string value contains a
    # ```json fence (e.g. a markdown field quoting fenced code). Extraction
    # must prefer whole-stdout JSON over the inner fence.
    stdout = json.dumps({"name": 'see ```json\n{"not": "the payload"}\n``` above', "count": 3})
    handler = SubprocessHandler()
    mocker.patch.object(handler, "_run", return_value=_fake_completed(stdout))
    result = handler.execute(prompt="extract", output_schema=Out, profile=_profile())
    assert isinstance(result.payload, Out)
    assert result.payload.count == 3


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
    with pytest.raises(ProfileCapabilityError, match="non-subprocess"):
        SubprocessHandler().execute(prompt="x", output_schema=Out, profile=bad_profile)


def test_arg_delivery_appends_prompt_to_argv(mocker: Any) -> None:
    """ARG delivery: the prompt is passed as the last positional argv."""
    handler = SubprocessHandler()
    seen: dict[str, Any] = {}

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        seen["argv"] = argv
        return _completed_bytes(argv)

    mocker.patch("chumak.handlers.subprocess.subprocess.run", side_effect=fake_run)
    handler.execute(
        prompt="extract",
        output_schema=Out,
        profile=_profile(delivery=PromptDelivery.ARG),
    )
    assert seen["argv"][0] == "claude"
    assert "extract" in seen["argv"][-1]


# --- Real process -----------------------------------------------------------

# A child that reports the SHA-256 of the exact bytes it received (on stdin,
# or as its last argument) as an `Out` payload.
_HASH_WHAT_ARRIVED = (
    "import hashlib, json, sys; "
    "data = sys.stdin.buffer.read() if sys.argv[1] == 'stdin' else sys.argv[2].encode('utf-8'); "
    "print(json.dumps({'name': hashlib.sha256(data).hexdigest(), 'count': len(data)}))"
)


@pytest.mark.parametrize("delivery", [PromptDelivery.STDIN, PromptDelivery.ARG])
def test_a_real_cli_receives_exactly_the_hashed_utf8(delivery: PromptDelivery, tmp_path) -> None:
    """Regression: stdin was a text-mode pipe in the locale's encoding. On
    Windows that's cp1252, which can't encode `✓` (the call hung until its
    timeout), and it writes `\n` as `\r\n`, so the CLI got bytes the prompt
    hash doesn't cover. The mocked tests above can't see either."""
    attachment = _swatch(tmp_path / "café ✓")
    profile = Profile(
        name="hash-what-arrived",
        handler=HandlerType.SUBPROCESS,
        model="python",
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(_HASH_WHAT_ARRIVED)} "
        f"{delivery.value}",
        prompt_delivery=delivery,
        timeout=20.0,
        attachment_ref="{path}",
    )

    result = infer(
        prompt="naïve → extract",
        attachments=[attachment],
        output_schema=Out,
        profile=profile,
    )

    assert result.payload.name == result.meta.produced_by.prompt_actual_sha256
