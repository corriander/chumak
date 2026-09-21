"""Live integration test for the System One (Jev) handler — **experimental**.

Hits the real TypeSafe endpoint, so it costs money (a fraction of a penny)
and needs a key. Skipped unless `--integration` is passed *and*
`CHUMAK_TEST_TYPESAFE_API_KEY` is set.

    CHUMAK_TEST_TYPESAFE_API_KEY=sk-... \\
        uv run pytest --integration tests/test_systemone_live.py -v

This is the ALS-95 exit condition in executable form: a validated payload
from a live call, with input and output tokens visible on `Meta`.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import chumak
from chumak.handlers.systemone import SystemOneRaw
from chumak.handlers.types import HandlerType

pytest.importorskip(
    "typesafe_sdk",
    reason="install with `uv sync --extra typesafe` to run this test",
)

API_KEY = os.environ.get("CHUMAK_TEST_TYPESAFE_API_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="set CHUMAK_TEST_TYPESAFE_API_KEY to run the live System One test",
)


class Triage(BaseModel):
    """The vendor's own documentation example, as a chumak schema."""

    department: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle this?",
        json_schema_extra={
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            }
        },
    )
    is_urgent: bool = Field(description="Does this convey urgency?")
    frustration: int = Field(
        description="How frustrated is the customer?",
        json_schema_extra={"criteria": ["Calm", "Frustrated", "Very angry"]},
    )


def _profile() -> chumak.Profile:
    model_kwargs: dict[str, object] = {"api_key": API_KEY}
    base_url = os.environ.get("CHUMAK_TEST_TYPESAFE_URL")
    if base_url:
        model_kwargs["base_url"] = base_url
    return chumak.Profile(
        name="jev",
        handler=HandlerType.SYSTEMONE,
        model=os.environ.get("CHUMAK_TEST_TYPESAFE_MODEL", "jev-latest"),
        model_kwargs=model_kwargs,
    )


@pytest.mark.integration
def test_systemone_returns_a_validated_payload_with_usage() -> None:
    result = chumak.infer(
        prompt="Help! My payouts have been failing for 3 days.",
        output_schema=Triage,
        profile=_profile(),
    )

    payload = result.payload
    assert isinstance(payload, Triage), f"got {type(payload).__name__}"
    # The documented example answer. Asserted because a decision model that
    # cannot route this one is not working at all.
    assert payload.department == "billing"
    assert payload.is_urgent is True
    assert 0 <= payload.frustration <= 2

    # ALS-95 exit condition: tokens visible on Meta.
    assert result.meta.cost.tokens_in is not None
    assert result.meta.cost.tokens_in > 0
    assert result.meta.cost.tokens_out is not None
    assert result.meta.produced_by.profile == "jev"
    # Cost is priced by the handler's own dated constants.
    assert result.meta.cost.usd is not None
    assert result.meta.cost.usd > 0


@pytest.mark.integration
def test_systemone_confidence_and_probabilities_are_available_on_raw() -> None:
    """Probabilities are the point of a decision model; prove they survive.

    `infer()` returns only payload/citations/meta, so this drives the
    handler directly to reach `raw`.
    """
    from chumak.handlers.systemone import SystemOneHandler

    handler = SystemOneHandler()
    result = handler.execute("Help! My payouts have been failing for 3 days.", Triage, _profile())

    raw = result.raw
    assert isinstance(raw, SystemOneRaw)
    assert raw.model.startswith("jev-")
    assert raw.confidence("department") is not None
    assert raw.answers["department"]["probabilities"]["billing"] > 0.5
    # Documented asymmetry: noul answers carry no confidence.
    assert raw.confidence("is_urgent") is None
    assert raw.estimated_usd() is not None
    assert raw.duration_ms is not None
    # The handle for reconciling this call against the vendor usage page.
    assert raw.request_id is not None
