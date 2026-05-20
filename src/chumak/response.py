"""Response types: validated payload, normalised citations, optional provenance.

`InferResult` is what `infer()` returns. `meta` is always present — model
identity and cost/citations are cheap to compute — but `meta.artefact_type` /
`artefact_id` / `derived_from` are populated only when the caller passes a
`Provenance` block.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Normalised citation. Shape mirrors LangChain's content-block
    citation annotations so LangChain-handler responses pass through
    largely unchanged.
    """

    url: str
    title: str | None = None
    quote: str | None = Field(
        default=None,
        description="Verbatim excerpt the model attributed to the source, if available.",
    )
    retrieved_at: datetime | None = None


class Cost(BaseModel):
    """Token usage and (where computable) USD cost for a single call.

    Best-effort: handler types that don't surface usage data leave fields
    as `None`. `usd` is not computed at the substrate level — that needs
    per-model pricing the library doesn't carry.
    """

    tokens_in: int | None = None
    tokens_out: int | None = None
    usd: float | None = None


class ArtefactRef(BaseModel):
    """Reference to an upstream artefact. `artefact_type` is versioned
    (e.g. `mission_title@v1`); `artefact_id` is the stable identifier
    within that type.
    """

    artefact_type: str
    artefact_id: str


class Provenance(BaseModel):
    """Caller-supplied provenance inputs. Pass on `infer()` if you want
    `Meta.artefact_type` / `artefact_id` / `derived_from` populated.
    """

    artefact_type: str
    artefact_id: str
    derived_from: list[ArtefactRef] = Field(default_factory=list)
    prompt_template_sha256: str | None = Field(
        default=None,
        description=(
            "SHA256 of the verbatim prompt-template file content. Optional — "
            "supply only if you load prompts from a versioned template file."
        ),
    )


class ProducedBy(BaseModel):
    """Identity of the inference that produced an artefact."""

    profile: str
    model: str = Field(
        description=(
            "Model identifier. Authoritative for langchain profiles "
            "(passed literally to `init_chat_model`); descriptive metadata "
            "for subprocess profiles (the profile's `command` pins the actual model)."
        ),
    )
    prompt_version: str | None = None
    prompt_template_sha256: str | None = None
    prompt_actual_sha256: str | None = Field(
        default=None,
        description=(
            "SHA256 of the actual text sent to the underlying transport, after "
            "any handler-level augmentation (e.g. subprocess schema injection). "
            "Differs from `prompt_template_sha256` by exactly the augmentation."
        ),
    )
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class Meta(BaseModel):
    """Inference-time observable metadata, plus optional provenance.

    `artefact_type` / `artefact_id` / `derived_from` are `None` / empty when
    the caller did not supply a `Provenance` block — keeping `meta` always
    populated avoids special-casing the no-provenance path.
    """

    artefact_type: str | None = None
    artefact_id: str | None = None
    derived_from: list[ArtefactRef] = Field(default_factory=list)
    produced_by: ProducedBy
    generated_at: datetime
    cost: Cost = Field(default_factory=Cost)
    citations: list[Citation] = Field(default_factory=list)


class InferResult(BaseModel):
    """Return shape from `infer()`. `payload` is an instance of the caller's
    `output_schema`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: Any
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations from the handler response. Also available on `meta.citations`.",
    )
    meta: Meta
