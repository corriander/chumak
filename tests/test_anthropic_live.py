"""Live integration test for the LangChain handler against Anthropic.

Proves image attachments survive the langchain-anthropic translation (standard
``image`` content blocks become Anthropic ``source`` blocks) against a real
model. The OpenAI-compatible route is covered by `test_langchain_live.py`.

Hits the paid API, so it is skipped unless `--integration` is passed *and*
`CHUMAK_TEST_ANTHROPIC_API_KEY` is set. It deliberately ignores
`ANTHROPIC_API_KEY`, so a key already in the environment can't spend money
by accident.

    uv sync --extra anthropic
    CHUMAK_TEST_ANTHROPIC_API_KEY=sk-ant-... \\
        uv run pytest --integration tests/test_anthropic_live.py -v

`CHUMAK_TEST_ANTHROPIC_MODEL` overrides the model (default a Haiku-class one).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

import chumak
from chumak.handlers.types import HandlerType
from tests.conftest import solid_png

pytest.importorskip(
    "langchain_anthropic",
    reason="install with `uv sync --extra anthropic` to run this test",
)

API_KEY = os.environ.get("CHUMAK_TEST_ANTHROPIC_API_KEY")
MODEL = os.environ.get("CHUMAK_TEST_ANTHROPIC_MODEL", "claude-haiku-4-5")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="set CHUMAK_TEST_ANTHROPIC_API_KEY to run the live Anthropic test",
)


class ColourTag(BaseModel):
    colour: str
    is_warm: bool


def _profile() -> chumak.Profile:
    return chumak.Profile(
        name="anthropic",
        handler=HandlerType.LANGCHAIN,
        model=f"anthropic:{MODEL}",
        temperature=0.0,
        max_tokens=256,
        api_key=API_KEY,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("rgb", "colour", "is_warm"),
    [((255, 0, 0), "red", True), ((0, 0, 255), "blue", False)],
    ids=["red", "blue"],
)
def test_image_attachment_through_anthropic(
    tmp_path: Path, rgb: tuple[int, int, int], colour: str, is_warm: bool
) -> None:
    """A flat-colour PNG attached to a typed call; the answer tracks the image.

    Red and blue together rule out a model that says "red" regardless of what
    it was sent.
    """
    path = tmp_path / f"{colour}.png"
    path.write_bytes(solid_png(16, 16, rgb))
    attachment = chumak.Attachment(path=path)

    result = chumak.infer(
        prompt=(
            "The attached image is a single flat colour. Set `colour` to its name "
            "(lowercase, one word) and `is_warm` to true for warm tones, false for cool."
        ),
        attachments=[attachment],
        output_schema=ColourTag,
        profile=_profile(),
    )

    payload = result.payload
    assert isinstance(payload, ColourTag), f"got {type(payload).__name__}"
    assert payload.colour.strip().lower() == colour
    assert payload.is_warm is is_warm

    assert result.meta.produced_by.model == f"anthropic:{MODEL}"
    assert [d.model_dump() for d in result.meta.produced_by.attachments] == [
        attachment.digest().model_dump()
    ]
