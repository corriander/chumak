"""Public surface: `infer()`.

A thin coordinator. Resolves the profile's handler, executes, builds meta,
returns the validated payload alongside citations and meta. Provenance is
opt-in — pass `provenance=Provenance(...)` to populate `meta.artefact_type`
/ `artefact_id` / `derived_from`.
"""

from __future__ import annotations

from pydantic import BaseModel

from chumak.handlers import HANDLER_REGISTRY
from chumak.meta import build_meta
from chumak.profile import Profile
from chumak.response import InferResult, Provenance


def infer(
    *,
    prompt: str,
    output_schema: type[BaseModel] | None = None,
    profile: Profile,
    provenance: Provenance | None = None,
) -> InferResult:
    """Run a single inference call.

    Arguments:
        prompt: Verbatim prompt text. The library does no templating.
        output_schema: Optional Pydantic `BaseModel` subclass. When given, the
            handler returns a validated instance of it in `result.payload`.
            When omitted (``None``), the call is untyped: `result.payload` is
            the model's plain response text, and no structured-output
            translation is applied. Not every handler supports the untyped
            path — the subprocess handler requires a schema (it injects the
            JSON schema into the prompt) and raises without one.
        profile: A loaded `Profile` (usually from `ProfileLoader.load`).
        provenance: Optional. Pass to stamp artefact identifiers and
            upstream references into `result.meta`.
    """
    handler_cls = HANDLER_REGISTRY[profile.handler]
    handler = handler_cls()
    handler_result = handler.execute(prompt=prompt, output_schema=output_schema, profile=profile)
    meta = build_meta(profile=profile, handler_result=handler_result, provenance=provenance)
    return InferResult(
        payload=handler_result.payload,
        citations=meta.citations,
        meta=meta,
    )
