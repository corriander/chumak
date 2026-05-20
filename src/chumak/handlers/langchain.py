"""LangChain-backed handler for SDK profiles.

`init_chat_model(profile.model)` collapses provider routing into a single
identifier (`"anthropic:..."`, `"openai:..."`, `"google_genai:..."`). The
structured-output translation, citation normalisation, and token-usage
surfacing are LangChain's job rather than ours — this handler is a thin
adapter.
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
        output_schema: type[BaseModel],
        profile: Profile,
    ) -> HandlerResult:
        kwargs: dict[str, Any] = {}
        if profile.temperature is not None:
            kwargs["temperature"] = profile.temperature
        if profile.max_tokens is not None:
            kwargs["max_tokens"] = profile.max_tokens
        kwargs.update(profile.model_kwargs)

        model = init_chat_model(profile.model, **kwargs)
        structured = model.with_structured_output(output_schema, include_raw=True)
        result = structured.invoke(prompt)

        if result.get("parsing_error"):
            raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
        return HandlerResult(
            payload=result["parsed"],
            raw=result["raw"],
            rendered_prompt=prompt,
        )
