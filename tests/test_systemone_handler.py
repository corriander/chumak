"""Unit tests for the experimental System One (Jev) handler.

Everything here runs against an `httpx2.MockTransport` — no network, no API
key. Driving the SDK's real client means the retry tests exercise the
vendor's actual `RetryPolicy` rather than a reimplementation of it.

The live counterpart is `test_systemone_live.py`, gated on `--integration`.
"""

from __future__ import annotations

import enum
from typing import Any, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field

import chumak
from chumak.attachments import Attachment
from chumak.errors import ProfileCapabilityError
from chumak.handlers.systemone import (
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


class Recorder:
    """A MockTransport plus the requests it saw."""

    def __init__(self, responses: list[tuple[int, Any]]) -> None:
        self._queue = list(responses)
        self.requests: list[httpx2.Request] = []
        self.transport = httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, payload = self._queue.pop(0)
        headers = {"x-typesafe-request-id": f"req_{len(self.requests)}"}
        if isinstance(payload, bytes):
            return httpx2.Response(status, content=payload, headers=headers)
        return httpx2.Response(status, json=payload, headers=headers)

    def body(self, index: int = 0) -> dict[str, Any]:
        import json

        return json.loads(self.requests[index].content)


def make_profile(*, api_key: str | None = "sk-test", **model_kwargs: Any) -> Profile:
    # Fast backoff so retry tests don't sleep for real.
    kwargs: dict[str, Any] = {
        "retry": {"backoff_initial": 0.001, "backoff_max": 0.002, "backoff_jitter": 0.0},
    }
    kwargs.update(model_kwargs)
    return Profile(
        name="jev",
        handler=HandlerType.SYSTEMONE,
        model="jev-latest",
        api_key=api_key,
        model_kwargs=kwargs,
    )


# --------------------------------------------------------------------------
# Schema -> questions (SDK-free; these must pass without the extra)
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
    rec = Recorder([(200, ok_response())])
    SystemOneHandler(rec.transport).execute("drawer text", Verdict, make_profile())

    body = rec.body()
    assert body["state"] == "drawer text"
    assert body["model"] == "jev-latest"
    assert set(body["questions"]) == {"room", "disposition", "is_settled"}
    assert body["questions"]["is_settled"]["type"] == "noul"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test"


def test_execute_returns_a_validated_payload() -> None:
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    payload = result.payload
    assert isinstance(payload, Verdict)
    assert payload.room == "architecture"
    assert payload.is_settled is True
    # Score is probability-weighted; 2.4 rounds to the int field's 2.
    assert payload.disposition == 2


def test_probabilities_and_confidence_stay_off_the_payload() -> None:
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    assert not hasattr(result.payload, "confidence")
    raw = result.raw
    assert isinstance(raw, SystemOneRaw)
    assert raw.confidence("room") == pytest.approx(0.66)
    assert raw.answers["room"]["probabilities"]["architecture"] == pytest.approx(0.7)


def test_noul_answers_carry_no_confidence() -> None:
    """Vendor API asymmetry, worth pinning: only choice/score have it."""
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.raw.confidence("is_settled") is None


def test_noul_threshold_is_configurable() -> None:
    response = ok_response()
    response["answers"]["is_settled"]["noul"] = 0.6
    result = SystemOneHandler(Recorder([(200, response)]).transport).execute(
        "drawer text", Verdict, make_profile(noul_threshold=0.75)
    )
    assert result.payload.is_settled is False


def test_rendered_prompt_is_the_state_actually_sent() -> None:
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.rendered_prompt == "drawer text"


def test_raw_never_carries_the_api_key() -> None:
    import json

    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    assert "sk-test" not in json.dumps(result.raw.questions)


def test_resolved_model_version_is_preferred_over_the_alias() -> None:
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "drawer text", Verdict, make_profile()
    )
    assert result.raw.model == "jev-1.13.0"


def test_unknown_model_kwargs_are_forwarded_as_extra_body() -> None:
    rec = Recorder([(200, ok_response())])
    SystemOneHandler(rec.transport).execute("s", Verdict, make_profile(future_flag=True))
    assert rec.body()["future_flag"] is True


def test_untyped_call_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires an output_schema"):
        SystemOneHandler(Recorder([]).transport).execute("s", None, make_profile())


def test_profile_api_key_is_sent_as_the_bearer_token() -> None:
    rec = Recorder([(200, ok_response())])
    SystemOneHandler(rec.transport).execute("s", Verdict, make_profile())
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test"


def test_unset_api_key_falls_back_to_the_sdk_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-from-sdk-env")
    rec = Recorder([(200, ok_response())])
    SystemOneHandler(rec.transport).execute("s", Verdict, make_profile(api_key=None))
    assert rec.requests[0].headers["authorization"] == "Bearer sk-from-sdk-env"


def test_no_key_anywhere_names_the_sdk_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemOneError, match="TYPESAFE_API_KEY"):
        SystemOneHandler(Recorder([]).transport).execute("s", Verdict, make_profile(api_key=None))


def test_non_mapping_retry_config_is_rejected() -> None:
    with pytest.raises(SystemOneError, match="retry must be a mapping"):
        SystemOneHandler(Recorder([]).transport).execute("s", Verdict, make_profile(retry="fast"))


# --------------------------------------------------------------------------
# Retry — now the SDK's policy, not ours
# --------------------------------------------------------------------------


def test_429_is_retried_by_the_sdk_then_succeeds() -> None:
    rec = Recorder([(429, {"error": "slow down"}), (200, ok_response())])
    result = SystemOneHandler(rec.transport).execute("s", Verdict, make_profile())

    assert isinstance(result.payload, Verdict)
    assert len(rec.requests) == 2


def test_500_is_retried_too_which_the_old_hand_rolled_loop_did_not_do() -> None:
    rec = Recorder([(500, {"error": "boom"}), (200, ok_response())])
    result = SystemOneHandler(rec.transport).execute("s", Verdict, make_profile())
    assert isinstance(result.payload, Verdict)
    assert len(rec.requests) == 2


def test_retries_are_exhausted_and_then_raise() -> None:
    rec = Recorder([(429, {"error": "slow down"})] * 6)
    with pytest.raises(SystemOneError, match="call failed"):
        SystemOneHandler(rec.transport).execute(
            "s", Verdict, make_profile(retry={"max_retries": 2, "backoff_initial": 0.001})
        )
    assert len(rec.requests) == 3  # 1 attempt + 2 retries


def test_max_retries_zero_disables_retry() -> None:
    rec = Recorder([(429, {}), (200, ok_response())])
    with pytest.raises(SystemOneError):
        SystemOneHandler(rec.transport).execute(
            "s", Verdict, make_profile(retry={"max_retries": 0})
        )
    assert len(rec.requests) == 1


def test_422_is_not_retried() -> None:
    rec = Recorder([(422, {"error": "bad question"})])
    with pytest.raises(SystemOneError, match="call failed"):
        SystemOneHandler(rec.transport).execute("s", Verdict, make_profile())
    assert len(rec.requests) == 1


def test_401_is_not_retried() -> None:
    rec = Recorder([(401, {"error": "invalid api key"})])
    with pytest.raises(SystemOneError):
        SystemOneHandler(rec.transport).execute("s", Verdict, make_profile())
    assert len(rec.requests) == 1


def test_missing_answer_for_a_question_is_reported() -> None:
    response = ok_response()
    del response["answers"]["room"]
    with pytest.raises(SystemOneError, match=r"omitted answers for: \['room'\]"):
        SystemOneHandler(Recorder([(200, response)]).transport).execute(
            "s", Verdict, make_profile()
        )


# --------------------------------------------------------------------------
# Usage, cost, and the meta builder
# --------------------------------------------------------------------------


def test_token_usage_and_usd_reach_meta_through_infer() -> None:
    """The ALS-96 dependency: tokens AND cost must arrive in `Meta`."""
    from chumak.handlers import HANDLER_REGISTRY

    rec = Recorder([(200, ok_response())])

    class _Bound(SystemOneHandler):
        """A no-arg handler class, which is what the registry stores."""

        def __init__(self) -> None:
            super().__init__(rec.transport)

    original = HANDLER_REGISTRY[HandlerType.SYSTEMONE]
    HANDLER_REGISTRY[HandlerType.SYSTEMONE] = _Bound
    try:
        result = chumak.infer(prompt="s", output_schema=Verdict, profile=make_profile())
    finally:
        HANDLER_REGISTRY[HandlerType.SYSTEMONE] = original

    assert result.meta.cost.tokens_in == 296
    assert result.meta.cost.tokens_out == 20
    assert result.meta.cost.usd == pytest.approx(296 / 1_000_000 * INPUT_USD_PER_MTOK)
    assert result.meta.produced_by.profile == "jev"


def test_estimated_usd_uses_the_dated_price_constant() -> None:
    raw = SystemOneRaw(
        model="jev-1.13.0",
        questions={},
        usage={"input_tokens": 1_000_000, "output_tokens": 500},
    )
    assert raw.estimated_usd() == pytest.approx(INPUT_USD_PER_MTOK)


def test_estimated_usd_is_none_without_usage() -> None:
    assert SystemOneRaw(model="m", questions={}).estimated_usd() is None


def test_request_id_is_captured_for_reconciliation() -> None:
    """The correlation handle for checking a call against the usage page."""
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "s", Verdict, make_profile()
    )
    assert result.raw.request_id == "req_1"


def test_a_response_without_a_request_id_header_still_succeeds() -> None:
    """`request_id` is a property that raises; a missing id must not fail the call."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=ok_response())

    result = SystemOneHandler(httpx2.MockTransport(handler)).execute("s", Verdict, make_profile())
    assert result.raw.request_id is None
    assert isinstance(result.payload, Verdict)


def test_duration_is_recorded_for_the_ledger() -> None:
    result = SystemOneHandler(Recorder([(200, ok_response())]).transport).execute(
        "s", Verdict, make_profile()
    )
    assert result.raw.duration_ms is not None
    assert result.raw.duration_ms >= 0


def test_attachments_are_rejected_before_any_request(red_png) -> None:
    """Jev takes a text state only: an image must refuse loudly, never be dropped."""
    recorder = Recorder([(200, ok_response())])
    with pytest.raises(ProfileCapabilityError, match="does not accept attachments"):
        SystemOneHandler(recorder.transport).execute(
            "drawer text", Verdict, make_profile(), attachments=[Attachment(path=red_png)]
        )
    assert recorder.requests == []
