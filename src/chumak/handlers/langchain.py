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

`resolve_model()` is the public half of this module: it constructs the chat
model a profile describes and hands it to the caller, for consumers that run
their own loop (agent graphs, multi-turn, streaming) but still want chumak to
answer "which model, configured how". `execute()` goes through the same
function, so there is exactly one kwargs-assembly path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from chumak.errors import ProfileCapabilityError
from chumak.handlers.base import HandlerResult
from chumak.handlers.types import HandlerType

if TYPE_CHECKING:
    from chumak.profile import Profile


def resolve_model(profile: Profile) -> BaseChatModel:
    """Construct the chat model a langchain-handler profile describes.

    Applies the profile's `temperature` / `max_tokens` / `api_key` and then
    `model_kwargs` (which win on conflict) and passes the lot to
    `init_chat_model`. This is the exact model `infer()` would call for the
    same profile.

    Raises `ProfileCapabilityError` for profiles on any other handler: a
    subprocess profile pins its model inside a CLI command and has no chat
    model to hand out. That partiality is the honest contract — model
    resolution is a langchain-handler capability, not a profile-wide one.
    """
    if profile.handler is not HandlerType.LANGCHAIN:
        raise ProfileCapabilityError(
            f"Profile {profile.name!r} uses the {profile.handler.value!r} handler; "
            f"only {HandlerType.LANGCHAIN.value!r} profiles resolve to a chat model"
        )
    kwargs: dict[str, Any] = {}
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature
    if profile.max_tokens is not None:
        kwargs["max_tokens"] = profile.max_tokens
    # Unwrapped only here, at the SDK boundary. Unset: the provider class
    # reads its own variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
    if profile.api_key is not None:
        kwargs["api_key"] = profile.api_key.get_secret_value()
    kwargs.update(profile.model_kwargs)
    return init_chat_model(profile.model, **kwargs)


class LangChainHandler:
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult:
        model = resolve_model(profile)

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
        if not isinstance(result, dict):
            # `include_raw=True` contractually yields the {parsed, raw, parsing_error}
            # envelope; anything else means the integration broke that contract.
            raise TypeError(
                f"Structured output returned {type(result).__name__}, expected include_raw dict"
            )

        if result.get("parsing_error"):
            raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
        return HandlerResult(
            payload=result["parsed"],
            raw=result["raw"],
            rendered_prompt=prompt,
        )
