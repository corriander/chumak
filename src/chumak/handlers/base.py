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

from typing import TYPE_CHECKING, Any, Protocol

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


class Handler(Protocol):
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult: ...
