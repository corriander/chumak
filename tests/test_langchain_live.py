"""Live integration test for the LangChain handler.

Exercises chumak's LangChain handler against an OpenAI-compatible
inference endpoint — typically a local llama.cpp server, vLLM, LiteLLM,
or any other OpenAI-API-compatible backend. The point is to validate the
`openai:<model>` + `base_url` wiring end-to-end before tagging 0.1.0,
without needing a paid API key.

Run with:

    # default: http://localhost:8080/v1, model gpt-3.5-turbo
    uv run pytest --integration tests/test_langchain_live.py -v

    # custom endpoint / model
    CHUMAK_TEST_OPENAI_URL=http://localhost:8000/v1 \\
    CHUMAK_TEST_OPENAI_MODEL=qwen2.5-7b-instruct \\
        uv run pytest --integration tests/test_langchain_live.py -v

The test installs `langchain-openai` on demand via the `openai` extra:

    uv sync --extra openai
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import BaseModel

import chumak
from chumak.handlers.types import HandlerType

pytest.importorskip(
    "langchain_openai",
    reason="install with `uv sync --extra openai` to run this test",
)


class ColourTag(BaseModel):
    """Tiny structured-output schema. Small models reliably emit this."""

    colour: str
    is_warm: bool


def _build_profile() -> chumak.Profile:
    url = os.environ.get("CHUMAK_TEST_OPENAI_URL", "http://localhost:8080/v1")
    model = os.environ.get("CHUMAK_TEST_OPENAI_MODEL", "gpt-3.5-turbo")
    api_key = os.environ.get("CHUMAK_TEST_OPENAI_API_KEY", "sk-no-auth")

    model_kwargs: dict[str, Any] = {
        "base_url": url,
        "api_key": api_key,
    }

    return chumak.Profile(
        name="local-openai",
        handler=HandlerType.LANGCHAIN,
        model=f"openai:{model}",
        temperature=0.0,
        model_kwargs=model_kwargs,
    )


@pytest.mark.integration
def test_langchain_handler_against_openai_compat() -> None:
    """End-to-end: chumak.infer through LangChain against a local OpenAI-compat server.

    Asks the model to classify a single colour. Local models tend to be
    fine on tiny structured-output tasks even at low parameter counts.
    """
    profile = _build_profile()

    result = chumak.infer(
        prompt=(
            "Classify the colour 'crimson'. Set `colour` to the name "
            "(lowercase) and `is_warm` to true for warm tones, false for cool."
        ),
        output_schema=ColourTag,
        profile=profile,
    )

    payload = result.payload
    assert isinstance(payload, ColourTag), f"got {type(payload).__name__}"
    assert payload.colour.strip().lower() == "crimson"
    assert payload.is_warm is True

    # Meta is always populated; produced_by mirrors the profile.
    assert result.meta.produced_by.profile == "local-openai"
    assert result.meta.produced_by.model.startswith("openai:")
    assert result.meta.generated_at is not None


@pytest.mark.integration
def test_langchain_handler_untyped_plaintext() -> None:
    """End-to-end untyped path: no output_schema, plain-text `.invoke()`.

    This is the shape a liveness/smoke probe or one-shot free-text question
    uses — it validates the endpoint and the model's free-text response
    without asking the model to satisfy a Pydantic schema.
    """
    profile = _build_profile()

    result = chumak.infer(
        prompt="Reply with the single word: pong.",
        profile=profile,
    )

    assert isinstance(result.payload, str)
    assert result.payload.strip() != ""
    assert result.meta.produced_by.profile == "local-openai"
    assert result.meta.generated_at is not None
