"""Meta builder.

Stamps a `Meta` instance from a profile, an optional `Provenance` block, and
whatever native response the handler produced. Cost and citation extraction
are best-effort: handler types that don't surface either yield empty values
rather than raising.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage

from chumak.handlers.base import HandlerResult
from chumak.profile import Profile
from chumak.response import Citation, Cost, Meta, ProducedBy, Provenance


def build_meta(
    *,
    profile: Profile,
    handler_result: HandlerResult,
    provenance: Provenance | None = None,
) -> Meta:
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
            prompt_actual_sha256=_sha256_or_none(handler_result.rendered_prompt),
            model_kwargs=dict(profile.model_kwargs),
        ),
        generated_at=datetime.now(UTC),
        cost=_extract_cost(handler_result.raw),
        citations=_extract_citations(handler_result.raw),
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
