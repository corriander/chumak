"""Handler protocol and the `HandlerResult` they return.

A handler executes an inference request against some transport (a LangChain
chat model, a CLI subprocess, …) and returns:

  - `payload`: the parsed `output_schema` instance, or — when the call is
    untyped (`output_schema is None`) — the model's plain response text.
  - `raw`: whatever native object the transport produced (for the meta
    builder to mine for cost / citations).
  - `rendered_prompt`: the exact text sent to the transport, after any
    handler-level augmentation (e.g. JSON Schema injection for subprocess
    handlers). The meta builder hashes this for
    `produced_by.prompt_actual_sha256`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from chumak.profile import Profile

# Re-exported here so handlers/surface can spell the output_schema type
# tightly without each module re-importing `pydantic.BaseModel`.
OutputSchema = type[BaseModel]


class HandlerResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: Any
    raw: Any = None
    rendered_prompt: str | None = Field(
        default=None,
        description=(
            "The actual text sent to the underlying transport, after any "
            "handler-level augmentation. Hashed for provenance."
        ),
    )


@runtime_checkable
class UsageSource(Protocol):
    """A handler `raw` that can report its own token usage.

    The meta builder needs tokens out of every transport, but it cannot
    grow an `isinstance` branch per handler without becoming the one place
    that knows about all of them. Instead a `raw` object opts in by
    implementing this method, and `build_meta` asks rather than inspects.

    LangChain's `AIMessage` predates this and is special-cased in
    `chumak.meta`; everything added since is expected to implement it.
    """

    def token_usage(self) -> tuple[int | None, int | None]:
        """Return `(tokens_in, tokens_out)`, either of which may be `None`."""
        ...


class Handler(Protocol):
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult: ...
