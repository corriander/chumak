"""Meta builder.

Stamps a `Meta` instance from a profile, an optional `Provenance` block, and
whatever native response the transport produced. Cost and citation extraction
are best-effort: handler types that don't surface either yield empty values
rather than raising.

`build_meta` is public. `infer()` uses it, and so can a consumer that made
the model call itself (via `resolve_model`) and wants the same envelope
stamped on the result — chumak records the call; it never runs the loop.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage

from chumak.handlers.base import UsageSource
from chumak.profile import Profile
from chumak.response import Citation, Cost, Meta, ProducedBy, Provenance


def build_meta(
    raw: Any,
    *,
    profile: Profile,
    prompt: str | None,
    provenance: Provenance | None = None,
) -> Meta:
    """Stamp the `Meta` / `ProducedBy` / `Cost` / `Citation` envelope for one call.

    Arguments:
        raw: The native response object. An `AIMessage` yields token usage
            and citations; anything else (e.g. the subprocess handler's
            `SubprocessRaw`) yields empty best-effort fields.
        profile: The `Profile` the call was made with. Supplies model
            identity for `produced_by`.
        prompt: The exact text sent to the transport. Hashed into
            `produced_by.prompt_actual_sha256`; never stored verbatim.
            `None` means the sent text is unknown, and records no hash.
        provenance: Optional artefact identifiers and upstream references.
    """
    template_sha = provenance.prompt_template_sha256 if provenance else None
    return Meta(
        artefact_type=provenance.artefact_type if provenance else None,
        artefact_id=provenance.artefact_id if provenance else None,
        derived_from=list(provenance.derived_from) if provenance else [],
        produced_by=ProducedBy(
            profile=profile.name,
            model=profile.model,
            prompt_version=profile.prompt_version,
            prompt_template_sha256=template_sha,
            prompt_actual_sha256=_sha256_or_none(prompt),
        ),
        generated_at=datetime.now(UTC),
        cost=_extract_cost(raw),
        citations=_extract_citations(raw),
    )


def _sha256_or_none(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_cost(raw: Any) -> Cost:
    if isinstance(raw, AIMessage):
        usage = raw.usage_metadata or {}
        return Cost(
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )
    if isinstance(raw, UsageSource):
        tokens_in, tokens_out = raw.token_usage()
        return Cost(tokens_in=tokens_in, tokens_out=tokens_out, usd=raw.estimated_usd())
    return Cost()


def _extract_citations(raw: Any) -> list[Citation]:
    if not isinstance(raw, AIMessage):
        return []
    blocks = getattr(raw, "content_blocks", None) or []
    out: list[Citation] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for ann in block.get("annotations", []) or []:
            if ann.get("type") == "citation" and ann.get("url"):
                out.append(
                    Citation(
                        url=ann["url"],
                        title=ann.get("title"),
                        quote=ann.get("quote"),
                    )
                )
    return out
