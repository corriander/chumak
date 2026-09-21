"""Unit tests for the experimental System One (Jev) handler.

Everything here runs against a fake transport — no network, no API key.
The live counterpart is `test_systemone_live.py`, gated on `--integration`.
"""

from __future__ import annotations

import enum
import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

import chumak
from chumak.handlers.systemone import (
    DEFAULT_ENDPOINT,
    INPUT_USD_PER_MTOK,
    SystemOneError,
    SystemOneHandler,
    SystemOneRaw,
    SystemOneSchemaError,
    questions_from_schema,
)
from chumak.handlers.types import HandlerType
from chumak.profile import Profile


class Room(enum.StrEnum):
    DECISIONS = "decisions"
    GENERAL = "general"


class Verdict(BaseModel):
    """Mirrors the shape the ALS-97 corpus trial wants."""

    room: Literal["technical", "architecture", "general"] = Field(
        description="Which room does this drawer belong in?",
        json_schema_extra={
            "criteria": {
                "technical": "Implementation detail, code, config",
                "architecture": "Structure and design rationale",
                "general": "Anything else",
            }
        },
    )
    disposition: int = Field(
        description="What should happen to this drawer?",
        json_schema_extra={"criteria": ["discard", "merge", "keep", "promote"]},
    )
    is_settled: bool = Field(
        description="Does this record a decision or settled fact, not exploration?"
    )


class FakeTransport:
    """Records calls and replays a queued list of `(status, body)`."""

    def __init__(self, responses: list[tuple[int, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: Any,
        body: bytes,
        timeout: float,
    ) -> tuple[int, bytes]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body),
                "timeout": timeout,
            }
        )
        status, payload = self._responses.pop(0)
        encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, encoded


def make_profile(**model_kwargs: Any) -> Profile:
    kwargs = {"api_key": "sk-test"} | model_kwargs
    return Profile(
        name="jev",
        handler=HandlerType.SYSTEMONE,
        model="jev-latest",
        model_kwargs=kwargs,
    )


def ok_response() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "room": {
                "type": "choice",
                "choice": "architecture",
                "probabilities": {"technical": 0.2, "architecture": 0.7, "general": 0.1},
                "confidence": 0.66,
            },
            "disposition": {
                "type": "score",
                "score": 2.4,
                "legend": {"0": "discard", "1": "merge", "2": "keep", "3": "promote"},
                "probabilities": {"0": 0.0, "1": 0.1, "2": 0.4, "3": 0.5},
                "confidence": 0.55,
            },
            "is_settled": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }


# --------------------------------------------------------------------------
# Schema -> questions
# --------------------------------------------------------------------------


def test_literal_field_becomes_a_choice_with_its_rubric() -> None:
    questions = questions_from_schema(Verdict)
    assert questions["room"]["type"] == "choice"
    assert questions["room"]["instructions"] == "Which room does this drawer belong in?"
    assert questions["room"]["criteria"]["architecture"] == "Structure and design rationale"


def test_bool_field_becomes_a_noul() -> None:
    assert questions_from_schema(Verdict)["is_settled"]["type"] == "noul"


def test_numeric_field_becomes_a_score_with_ordered_levels() -> None:
    score = questions_from_schema(Verdict)["disposition"]
    assert score["type"] == "score"
    assert score["criteria"] == ["discard", "merge", "keep", "promote"]


def test_enum_field_becomes_a_choice_over_its_values() -> None:
    class M(BaseModel):
        room: Room = Field(description="Which room?")

    assert set(questions_from_schema(M)["room"]["criteria"]) == {"decisions", "general"}


def test_choice_without_a_rubric_offers_bare_options() -> None:
    class M(BaseModel):
        pick: Literal["a", "b"] = Field(description="Pick one")

    assert questions_from_schema(M)["pick"]["criteria"] == {"a": None, "b": None}


def test_optional_field_is_unwrapped_to_its_inner_type() -> None:
    class M(BaseModel):
        flag: bool | None = Field(default=None, description="Is it so?")

    assert questions_from_schema(M)["flag"]["type"] == "noul"


def test_missing_description_falls_back_to_the_field_name() -> None:
    class M(BaseModel):
        is_urgent: bool

    assert questions_from_schema(M)["is_urgent"]["instructions"] == "is urgent"


def test_numeric_field_without_a_rubric_is_rejected() -> None:
    class M(BaseModel):
        rating: int = Field(description="How good?")

    with pytest.raises(SystemOneSchemaError, match="ordered rubric"):
        questions_from_schema(M)


def test_score_rejects_more_levels_than_the_api_accepts() -> None:
    class M(BaseModel):
        rating: int = Field(
            description="How good?",
            json_schema_extra={"criteria": [str(i) for i in range(11)]},
        )

    with pytest.raises(SystemOneSchemaError, match="2-10 levels"):
        questions_from_schema(M)


def test_criteria_naming_an_unknown_option_is_rejected() -> None:
    class M(BaseModel):
        pick: Literal["a"] = Field(
            description="Pick", json_schema_extra={"criteria": {"a": "A", "z": "nope"}}
        )

    with pytest.raises(SystemOneSchemaError, match="not in the annotation"):
        questions_from_schema(M)


def test_unsupported_field_type_is_rejected_by_name() -> None:
    class M(BaseModel):
        title: str = Field(description="What is it called?")

    with pytest.raises(SystemOneSchemaError, match="no System One question type fits"):
        questions_from_schema(M)


def test_schema_with_no_fields_is_rejected() -> None:
    class M(BaseModel):
        pass

    with pytest.raises(SystemOneSchemaError, match="no fields"):
        questions_from_schema(M)


# --------------------------------------------------------------------------
# execute()
# --------------------------------------------------------------------------


def test_execute_builds_the_documented_request_shape() -> None:
    transport = FakeTransport([(200, ok_response())])
    SystemOneHandler(transport).execute("drawer text", Verdict, make_profile())

    call = transport.calls[0]
    assert call["url"] == DEFAULT_ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["body"]["state"] == "drawer text"
    assert call["body"]["model"] == "jev-latest"
    assert set(call["body"]["questions"]) == {"room", "disposition", "is_settled"}


def test_execute_returns_a_validated_payload() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    payload = result.payload
    assert isinstance(payload, Verdict)
    assert payload.room == "architecture"
    assert payload.is_settled is True
    # Score is probability-weighted; 2.4 rounds to the int field's 2.
    assert payload.disposition == 2


def test_probabilities_and_confidence_stay_off_the_payload() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    assert not hasattr(result.payload, "confidence")
    raw = result.raw
    assert isinstance(raw, SystemOneRaw)
    assert raw.confidence("room") == pytest.approx(0.66)
    assert raw.answers["room"]["probabilities"]["architecture"] == pytest.approx(0.7)


def test_noul_answers_carry_no_confidence() -> None:
    """Documented API asymmetry, worth pinning: only choice/score have it."""
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.raw.confidence("is_settled") is None


def test_noul_threshold_is_configurable() -> None:
    response = ok_response()
    response["answers"]["is_settled"]["noul"] = 0.6
    result = SystemOneHandler(FakeTransport([(200, response)])).execute(
        "drawer text", Verdict, make_profile(noul_threshold=0.75)
    )
    assert result.payload.is_settled is False


def test_rendered_prompt_is_the_state_actually_sent() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.rendered_prompt == "drawer text"


def test_raw_never_carries_the_api_key() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    assert "sk-test" not in json.dumps(result.raw.request)


def test_resolved_model_version_is_preferred_over_the_alias() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.raw.model == "jev-1.13.0"


def test_base_url_override_is_honoured() -> None:
    transport = FakeTransport([(200, ok_response())])
    SystemOneHandler(transport).execute(
        "s", Verdict, make_profile(base_url="http://localhost:9999/v1/systemone")
    )
    assert transport.calls[0]["url"] == "http://localhost:9999/v1/systemone"


def test_unknown_model_kwargs_are_forwarded_as_top_level_body_fields() -> None:
    transport = FakeTransport([(200, ok_response())])
    SystemOneHandler(transport).execute("s", Verdict, make_profile(future_flag=True))
    assert transport.calls[0]["body"]["future_flag"] is True


def test_untyped_call_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an output_schema"):
        SystemOneHandler(FakeTransport([])).execute("s", None, make_profile())


def test_missing_api_key_is_rejected_with_the_env_path() -> None:
    profile = Profile(name="jev", handler=HandlerType.SYSTEMONE, model="jev-latest")
    with pytest.raises(SystemOneError, match="MODEL_KWARGS__API_KEY"):
        SystemOneHandler(FakeTransport([])).execute("s", Verdict, profile)


# --------------------------------------------------------------------------
# Errors and retry
# --------------------------------------------------------------------------


def test_429_is_retried_then_succeeds() -> None:
    slept: list[float] = []
    transport = FakeTransport([(429, {"error": "slow down"}), (200, ok_response())])
    handler = SystemOneHandler(transport, sleep=slept.append)

    result = handler.execute("s", Verdict, make_profile())

    assert isinstance(result.payload, Verdict)
    assert len(transport.calls) == 2
    assert result.raw.attempts == 2
    assert slept == [0.5]


def test_529_backs_off_exponentially_and_gives_up() -> None:
    slept: list[float] = []
    transport = FakeTransport([(529, {"error": "overloaded"})] * 4)
    handler = SystemOneHandler(transport, sleep=slept.append)

    with pytest.raises(SystemOneError, match="HTTP 529"):
        handler.execute("s", Verdict, make_profile())

    assert len(transport.calls) == 4  # 1 attempt + 3 retries
    assert slept == [0.5, 1.0, 2.0]


def test_422_is_not_retried() -> None:
    transport = FakeTransport([(422, {"error": "bad question"})])
    with pytest.raises(SystemOneError, match="HTTP 422"):
        SystemOneHandler(transport, sleep=lambda _: None).execute("s", Verdict, make_profile())
    assert len(transport.calls) == 1


def test_401_surfaces_the_server_message() -> None:
    transport = FakeTransport([(401, {"error": "invalid api key"})])
    with pytest.raises(SystemOneError, match="invalid api key"):
        SystemOneHandler(transport, sleep=lambda _: None).execute("s", Verdict, make_profile())


def test_max_retries_is_configurable() -> None:
    transport = FakeTransport([(429, {})] * 2)
    handler = SystemOneHandler(transport, sleep=lambda _: None)
    with pytest.raises(SystemOneError):
        handler.execute("s", Verdict, make_profile(max_retries=1))
    assert len(transport.calls) == 2


def test_non_json_response_is_reported_clearly() -> None:
    transport = FakeTransport([(200, b"<html>502 Bad Gateway</html>")])
    with pytest.raises(SystemOneError, match="non-JSON"):
        SystemOneHandler(transport).execute("s", Verdict, make_profile())


def test_missing_answer_for_a_question_is_reported() -> None:
    response = ok_response()
    del response["answers"]["room"]
    with pytest.raises(SystemOneError, match=r"omitted answers for: \['room'\]"):
        SystemOneHandler(FakeTransport([(200, response)])).execute("s", Verdict, make_profile())


def test_answer_of_the_wrong_shape_is_reported() -> None:
    response = ok_response()
    response["answers"]["room"] = {"type": "choice"}
    with pytest.raises(SystemOneError, match="no `choice` string"):
        SystemOneHandler(FakeTransport([(200, response)])).execute("s", Verdict, make_profile())


# --------------------------------------------------------------------------
# Usage, cost, and the meta builder
# --------------------------------------------------------------------------


def test_token_usage_reaches_meta_through_infer() -> None:
    """The ALS-96 dependency: tokens must arrive in `Meta` for every handler."""
    from chumak.handlers import HANDLER_REGISTRY

    class _Bound(SystemOneHandler):
        """A no-arg handler class, which is what the registry stores."""

        def __init__(self) -> None:
            super().__init__(FakeTransport([(200, ok_response())]))

    original = HANDLER_REGISTRY[HandlerType.SYSTEMONE]
    HANDLER_REGISTRY[HandlerType.SYSTEMONE] = _Bound
    try:
        result = chumak.infer(prompt="s", output_schema=Verdict, profile=make_profile())
    finally:
        HANDLER_REGISTRY[HandlerType.SYSTEMONE] = original

    assert result.meta.cost.tokens_in == 296
    assert result.meta.cost.tokens_out == 20
    assert result.meta.produced_by.profile == "jev"


def test_estimated_usd_uses_the_dated_price_constant() -> None:
    raw = SystemOneRaw(
        endpoint=DEFAULT_ENDPOINT,
        model="jev-1.13.0",
        request={},
        response={},
        usage={"input_tokens": 1_000_000, "output_tokens": 500},
    )
    assert raw.estimated_usd() == pytest.approx(INPUT_USD_PER_MTOK)


def test_estimated_usd_is_none_without_usage() -> None:
    raw = SystemOneRaw(endpoint=DEFAULT_ENDPOINT, model="m", request={}, response={})
    assert raw.estimated_usd() is None


def test_duration_is_recorded_for_the_ledger() -> None:
    result = SystemOneHandler(FakeTransport([(200, ok_response())])).execute(
        "s", Verdict, make_profile()
    )
    assert result.raw.duration_ms is not None
    assert result.raw.duration_ms >= 0
