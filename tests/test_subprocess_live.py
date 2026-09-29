"""Live integration test for subprocess attachments.

Runs a real vision-capable CLI through `infer()` with an `attachment_ref`
profile, so the CLI has to find and read each image from the path chumak
writes into the prompt. Skipped unless `--integration` is passed and
`CHUMAK_TEST_SUBPROCESS_VISION_COMMAND` is set:

    CHUMAK_TEST_SUBPROCESS_VISION_COMMAND="ollama run llava:latest --format {schema}" \\
        uv run pytest --integration tests/test_subprocess_live.py -v

`{schema}` in the command, if present, becomes the test schema's JSON Schema,
quoted as one argument. Ollama then constrains the reply to it; with plain
`--format json`, llava often echoes the schema's `properties` wrapper back,
and the call fails validation. `CHUMAK_TEST_SUBPROCESS_ATTACHMENT_REF` sets
the reference template (default `{path}`, which `ollama run` picks up). The
prompt goes in on stdin. For Claude Code:

    CHUMAK_TEST_SUBPROCESS_VISION_COMMAND="claude -p --model haiku" \\
    CHUMAK_TEST_SUBPROCESS_ATTACHMENT_REF="@{path}" \\
        uv run pytest --integration tests/test_subprocess_live.py -v

Each colour gets its own call, and neither the prompt nor the path names the
colour. Blind, llava answers "red", so green and blue are what show it read
the image; `claude -p` says it sees no image.
"""

from __future__ import annotations

import json
import os
import shlex

import pytest
from pydantic import BaseModel

import chumak
from chumak.handlers.types import HandlerType, PromptDelivery
from tests.conftest import solid_png

_COMMAND = os.environ.get("CHUMAK_TEST_SUBPROCESS_VISION_COMMAND")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _COMMAND, reason="set CHUMAK_TEST_SUBPROCESS_VISION_COMMAND to a vision CLI"
    ),
]


class Colour(BaseModel):
    colour: str


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [((255, 0, 0), "red"), ((0, 160, 0), "green"), ((0, 0, 255), "blue")],
    ids=["a", "b", "c"],  # the ids end up in pytest's paths, so they can't name the colour
)
def test_cli_reads_the_referenced_image(
    rgb: tuple[int, int, int], expected: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("swatch") / "swatch.png"
    path.write_bytes(solid_png(64, 64, rgb))
    attachment = chumak.Attachment(path=path)
    assert _COMMAND is not None
    schema = shlex.quote(json.dumps(Colour.model_json_schema()))
    profile = chumak.Profile(
        name="vision-cli",
        handler=HandlerType.SUBPROCESS,
        model="vision-cli",
        command=_COMMAND.replace("{schema}", schema),
        prompt_delivery=PromptDelivery.STDIN,
        timeout=300.0,
        attachment_ref=os.environ.get("CHUMAK_TEST_SUBPROCESS_ATTACHMENT_REF", "{path}"),
    )

    result = chumak.infer(
        prompt="What single colour fills this image? Answer with one lowercase colour word.",
        attachments=[attachment],
        output_schema=Colour,
        profile=profile,
    )

    assert expected in result.payload.colour.lower()
    assert result.meta.produced_by.attachments_referenced == [attachment.digest()]
    assert result.meta.produced_by.attachments == []
