"""LangChain-backed handler for SDK profiles.

`init_chat_model(profile.model)` collapses provider routing into a single
identifier (`"anthropic:..."`, `"openai:..."`, `"google_genai:..."`). The
structured-output translation, citation normalisation, and token-usage
surfacing are LangChain's job rather than ours — this handler is a thin
adapter.

Two paths, chosen by whether the caller supplies an `output_schema`:

  - **typed** (schema given): `.with_structured_output()` returns a validated
    instance of the schema in `payload`.
  - **untyped** (schema `None`): a plain `.invoke()` whose response text is
    returned in `payload`. No structured-output translation is applied, so
    the call tests only the endpoint and the model's free-text response — not
    the model's ability to satisfy a Pydantic schema. This is the path a
    liveness/smoke probe or a one-shot free-text question wants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from chumak.handlers.base import HandlerResult

if TYPE_CHECKING:
    from chumak.profile import Profile


class LangChainHandler:
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult:
        kwargs: dict[str, Any] = {}
        if profile.temperature is not None:
            kwargs["temperature"] = profile.temperature
        if profile.max_tokens is not None:
            kwargs["max_tokens"] = profile.max_tokens
        kwargs.update(profile.model_kwargs)

        model = init_chat_model(profile.model, **kwargs)

        if output_schema is None:
            # Untyped: plain generation, no structured-output translation.
            # `.text` collapses string-or-content-block responses to text.
            raw = model.invoke(prompt)
            return HandlerResult(
                payload=raw.text,
                raw=raw,
                rendered_prompt=prompt,
            )

        structured = model.with_structured_output(output_schema, include_raw=True)
        result = structured.invoke(prompt)

        if result.get("parsing_error"):
            raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
        return HandlerResult(
            payload=result["parsed"],
            raw=result["raw"],
            rendered_prompt=prompt,
        )
