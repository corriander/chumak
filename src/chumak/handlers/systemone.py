"""TypeSafe "System One" (Jev) handler — **experimental**.

Jev is a non-autoregressive decision model: it takes a text `state` plus a
map of typed *questions* and returns one typed *answer* each, with
calibrated probabilities, in a single parallel pass. It generates no text.

The contract mismatch this handler resolves
-------------------------------------------
chumak's surface speaks `prompt + output_schema`; Jev speaks
`state + questions`. We bridge by **deriving the question map from the
schema** (ALS-95 option A), so consumers keep calling `infer()` unchanged:

  - `prompt`                      -> `state`
  - each `output_schema` field    -> one question, keyed by the field name
  - each answer                   -> that field's value on the payload

Field type            Question   Criteria source
--------------------  ---------  --------------------------------------------
`bool`                `noul`     optional `{"true": ..., "false": ...}`
`Literal[...]`        `choice`   option -> rubric map (max 255 options)
`enum.Enum`           `choice`   option -> rubric map (max 255 options)
`int` / `float`       `score`    ordered list of 2-10 level descriptions

`instructions` comes from the field's `description`. Criteria come from
`Field(json_schema_extra={"criteria": ...})`; for Choice they default to
"every option, no rubric" when omitted, and for Score they are **required**
(there is no sane way to invent an ordered rubric).

Probabilities, per-option distributions and `confidence` are deliberately
*not* on the payload — the payload is a plain validated instance of the
caller's schema. They ride in `raw` (a `SystemOneRaw`), alongside token
usage and estimated cost, which the meta builder mines via the
`UsageSource` protocol.

Untyped calls are rejected: without a schema there are no questions to ask,
so there is nothing for Jev to answer. This mirrors the subprocess handler.

Transport
---------
Calls go through the vendor's own `typesafe-sdk` (optional extra
``chumak[typesafe]``). Retry/backoff is **the SDK's job, not ours**: its
`RetryPolicy` honours `Retry-After`, applies jitter, and retries connection
and timeout errors — none of which a hand-rolled status-code loop does, and
all of which matter when a consumer walks a corpus in a tight loop. The
`transport` seam is kept so unit tests can drive an `httpx2.MockTransport`
without a network or a key.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from chumak.errors import ProfileCapabilityError
from chumak.handlers.base import HandlerResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chumak.attachments import Attachment
    from chumak.profile import Profile

try:  # pragma: no cover - trivial import guard
    from typesafe_sdk import (
        Choice,
        ChoiceAnswer,
        Noul,
        NoulAnswer,
        NoulCriteria,
        RetryPolicy,
        Score,
        ScoreAnswer,
        TypeSafeClient,
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _SDK_AVAILABLE = False

# Price snapshot. Output tokens are free at this tier. Dated deliberately:
# this is a vendor price, and a stale constant that silently under-reports
# spend is worse than no constant at all. Cross-check against the TypeSafe
# usage page when reconciling (ALS-96).
PRICE_SNAPSHOT_DATE = "2026-09-19"
INPUT_USD_PER_MTOK = 0.042
OUTPUT_USD_PER_MTOK = 0.0

# Per the API reference: Choice accepts at most 255 options, Score 2-10 levels.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# A noul answer is a probability, not a boolean. Collapsing it to the
# schema's `bool` needs a cut point; 0.5 is the neutral one. Callers who
# care should read `raw` for the probability rather than move this.
_DEFAULT_NOUL_THRESHOLD = 0.5

_INSTALL_HINT = (
    "the System One handler needs the vendor SDK: `uv sync --extra typesafe` "
    "(or `pip install 'chumak[typesafe]'`)"
)


class SystemOneError(RuntimeError):
    """A System One call failed, or its response could not be interpreted."""


class SystemOneSchemaError(SystemOneError):
    """An `output_schema` field could not be expressed as a Jev question."""


@dataclass
class SystemOneRaw:
    """Native System One exchange, kept for provenance and cost.

    `answers` holds the full per-question response including probability
    distributions and `confidence` — everything the payload drops.
    """

    model: str
    questions: dict[str, Any]
    answers: dict[str, Any] = dc_field(default_factory=dict)
    usage: dict[str, Any] = dc_field(default_factory=dict)
    request_id: str | None = None
    duration_ms: float | None = None

    def token_usage(self) -> tuple[int | None, int | None]:
        """Satisfy `chumak.handlers.base.UsageSource` for the meta builder."""
        tokens_in = self.usage.get("input_tokens")
        tokens_out = self.usage.get("output_tokens")
        return (
            tokens_in if isinstance(tokens_in, int) else None,
            tokens_out if isinstance(tokens_out, int) else None,
        )

    def estimated_usd(self) -> float | None:
        """USD for this call, at the dated price constants above.

        `None` when the response carried no usage block. This is an
        *estimate* against a snapshotted price — TypeSafe's usage page is
        the billing source of truth, and `request_id` is how you reconcile
        an individual call against it.
        """
        tokens_in, tokens_out = self.token_usage()
        if tokens_in is None and tokens_out is None:
            return None
        return (tokens_in or 0) / 1_000_000 * INPUT_USD_PER_MTOK + (
            tokens_out or 0
        ) / 1_000_000 * OUTPUT_USD_PER_MTOK

    def confidence(self, question_id: str) -> float | None:
        """Confidence for one answer, where the answer type carries one.

        Choice and Score answers carry `confidence`; **noul answers do not**
        — for those the probability in `noul` is the whole signal. This is
        the vendor's own type shape, not a chumak limitation.
        """
        answer = self.answers.get(question_id)
        if not isinstance(answer, dict):
            return None
        value = answer.get("confidence")
        return float(value) if isinstance(value, int | float) else None


def _criteria_from_field(info: FieldInfo) -> Any:
    """Pull a `criteria` override out of `Field(json_schema_extra=...)`.

    `json_schema_extra` may also be a callable that mutates a schema dict;
    that form carries nothing for us to read, so it is treated as absent.
    """
    extra = info.json_schema_extra
    if isinstance(extra, dict):
        return extra.get("criteria")
    return None


def _instructions_from_field(name: str, info: FieldInfo) -> str:
    """A field's `description` is its question. Fall back to the field name.

    The fallback keeps a schema without descriptions *working*, but a bare
    field name is a poor question — Jev is being asked to decide something,
    and `disposition` alone says far less than a sentence would.
    """
    description = (info.description or "").strip()
    return description or name.replace("_", " ")


def _unwrap_optional(annotation: Any) -> Any:
    """Reduce `X | None` to `X`; leave every other annotation alone."""
    args = [a for a in get_args(annotation) if a is not type(None)]
    if (
        get_origin(annotation) is not None
        and len(args) == 1
        and type(None) in get_args(annotation)
    ):
        return args[0]
    return annotation


def _choice_options(annotation: Any) -> list[str] | None:
    """Options for a Choice, or `None` if this annotation is not one.

    Handles `Literal[...]` and `enum.Enum` subclasses. Enum *values* are
    used, not member names — the value is what a schema validates against.
    """
    if get_origin(annotation) is Literal:
        return [str(arg) for arg in get_args(annotation)]
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return [str(member.value) for member in annotation]
    return None


def _build_choice(name: str, info: FieldInfo, options: list[str]) -> dict[str, Any]:
    if len(options) > MAX_CHOICE_OPTIONS:
        raise SystemOneSchemaError(
            f"Field {name!r}: Choice accepts at most {MAX_CHOICE_OPTIONS} options, "
            f"got {len(options)}"
        )
    override = _criteria_from_field(info)
    if override is None:
        # No rubric supplied: offer every option bare. The API allows a null
        # description per option precisely for this case.
        criteria: dict[str, Any] = dict.fromkeys(options)
    elif isinstance(override, dict):
        unknown = sorted(str(key) for key in override if str(key) not in set(options))
        if unknown:
            raise SystemOneSchemaError(
                f"Field {name!r}: criteria name options not in the annotation: {unknown}"
            )
        criteria = {option: override.get(option) for option in options}
    else:
        raise SystemOneSchemaError(
            f"Field {name!r}: Choice criteria must be a mapping of option -> rubric, "
            f"got {type(override).__name__}"
        )
    return {
        "type": "choice",
        "instructions": _instructions_from_field(name, info),
        "criteria": criteria,
    }


def _build_noul(name: str, info: FieldInfo) -> dict[str, Any]:
    question: dict[str, Any] = {
        "type": "noul",
        "instructions": _instructions_from_field(name, info),
    }
    override = _criteria_from_field(info)
    if override is None:
        return question
    if not isinstance(override, dict) or not set(override) <= {"true", "false"}:
        raise SystemOneSchemaError(
            f"Field {name!r}: noul criteria must be a mapping with keys 'true' and/or 'false'"
        )
    question["criteria"] = override
    return question


def _build_score(name: str, info: FieldInfo) -> dict[str, Any]:
    override = _criteria_from_field(info)
    if override is None:
        raise SystemOneSchemaError(
            f"Field {name!r}: a numeric field needs an ordered rubric to become a Score. "
            f'Supply Field(json_schema_extra={{"criteria": ["worst", ..., "best"]}}).'
        )
    if not isinstance(override, list | tuple):
        raise SystemOneSchemaError(
            f"Field {name!r}: Score criteria must be an ordered sequence of level "
            f"descriptions, got {type(override).__name__}"
        )
    levels = list(override)
    if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise SystemOneSchemaError(
            f"Field {name!r}: Score accepts {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, "
            f"got {len(levels)}"
        )
    return {
        "type": "score",
        "instructions": _instructions_from_field(name, info),
        "criteria": levels,
    }


def questions_from_schema(output_schema: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Translate a Pydantic schema into a Jev `questions` map, as plain dicts.

    Public, and deliberately SDK-free: the question map is the reusable
    artefact of an experiment. Being able to inspect, diff and archive
    exactly what was asked matters more than the handler's plumbing, and
    should not require the optional extra to be installed.

    Question ids are field names, which is also how answers are keyed on
    the way back.
    """
    if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
        raise TypeError("output_schema must be a Pydantic BaseModel subclass")

    questions: dict[str, dict[str, Any]] = {}
    for name, info in output_schema.model_fields.items():
        annotation = _unwrap_optional(info.annotation)
        options = _choice_options(annotation)
        if options is not None:
            questions[name] = _build_choice(name, info, options)
        elif annotation is bool:
            questions[name] = _build_noul(name, info)
        elif annotation in (int, float):
            questions[name] = _build_score(name, info)
        else:
            raise SystemOneSchemaError(
                f"Field {name!r}: no System One question type fits {annotation!r}. "
                "Supported: bool (noul), Literal/Enum (choice), int/float (score)."
            )
    if not questions:
        raise SystemOneSchemaError(
            f"{output_schema.__name__} declares no fields, so there is nothing to ask"
        )
    return questions


def _to_sdk_questions(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lift the plain question dicts into the SDK's typed question objects."""
    built: dict[str, Any] = {}
    for name, question in questions.items():
        kind = question["type"]
        instructions = question["instructions"]
        if kind == "noul":
            # NoulCriteria is a TypedDict, so its keys are built explicitly
            # rather than unpacked — `_build_noul` has already rejected any
            # key outside {"true", "false"}.
            raw_criteria = question.get("criteria") or {}
            criteria: NoulCriteria = {}
            if "true" in raw_criteria:
                criteria["true"] = raw_criteria["true"]
            if "false" in raw_criteria:
                criteria["false"] = raw_criteria["false"]
            built[name] = Noul(
                instructions=instructions,
                criteria=criteria or None,
            )
        elif kind == "choice":
            built[name] = Choice(instructions=instructions, criteria=question["criteria"])
        else:
            built[name] = Score(instructions=instructions, criteria=question["criteria"])
    return built


def _value_from_answer(name: str, answer: Any, annotation: Any, *, noul_threshold: float) -> Any:
    """Collapse one typed SDK answer to the plain value the schema wants."""
    if isinstance(answer, NoulAnswer):
        return bool(answer.noul >= noul_threshold)
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, ScoreAnswer):
        # Score is probability-weighted and lands *between* levels, so an
        # int-annotated field has to be rounded rather than truncated.
        return round(answer.score) if annotation is int else float(answer.score)
    raise SystemOneError(f"Answer for {name!r} has unexpected type {type(answer).__name__}")


def _request_id_or_none(response: Any) -> str | None:
    """Read the SDK's `request_id`, tolerating its absence.

    It is a `cached_property` that *raises* when the response carried no
    `x-typesafe-request-id` header, so `getattr(..., None)` does not help:
    the default only covers a missing attribute, never one whose getter
    throws. A missing correlation id must not fail an otherwise good call.
    """
    try:
        return response.request_id
    except Exception:
        return None


class SystemOneHandler:
    """Experimental handler for TypeSafe System One (Jev).

    The key is `profile.api_key` (env overlay `{APP}_PROFILE_{NAME}_API_KEY`);
    left unset, the SDK reads `TYPESAFE_API_KEY` itself. It is never echoed
    into `raw`. Everything else rides `profile.model_kwargs` rather than new
    `Profile` fields — the same route `test_langchain_live.py` uses for
    `base_url`:

      - `base_url`       (default: the SDK's own)
      - `timeout`        (seconds)
      - `noul_threshold` (default 0.5)
      - `retry`          (mapping splatted into the SDK's `RetryPolicy`,
                          e.g. `{"max_retries": 5, "backoff_max": 10.0}`)

    Anything else is forwarded to the API as `extra_body`, so a new
    top-level request field does not require a chumak release to reach.
    """

    def __init__(self, transport: Any = None) -> None:
        # An `httpx2.BaseTransport`. Present so unit tests can drive a
        # MockTransport through the SDK's real retry path.
        self._transport = transport

    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
        attachments: Sequence[Attachment] = (),
    ) -> HandlerResult:
        if attachments:
            # Jev's input is a text `state`; there is nowhere to put an image.
            # Refuse rather than answer questions about a picture it never saw.
            raise ProfileCapabilityError(
                f"SystemOneHandler does not accept attachments (profile {profile.name!r}, "
                f"{len(attachments)} given): System One takes a text state only; "
                "attachments are a langchain-handler capability"
            )
        if not _SDK_AVAILABLE:
            raise SystemOneError(f"Profile {profile.name!r}: {_INSTALL_HINT}")
        if output_schema is None:
            # Jev answers questions; questions come from the schema. With no
            # schema there is no question to ask, and Jev emits no free text
            # that an untyped call could return. Mirrors SubprocessHandler.
            raise ValueError(
                f"SystemOneHandler requires an output_schema (profile {profile.name!r}): "
                "System One answers typed questions and generates no text, so there is "
                "no untyped path"
            )

        options = dict(profile.model_kwargs)
        base_url = options.pop("base_url", None)
        timeout = options.pop("timeout", None)
        noul_threshold = float(options.pop("noul_threshold", _DEFAULT_NOUL_THRESHOLD))
        retry_config = options.pop("retry", None)
        if retry_config is not None and not isinstance(retry_config, dict):
            raise SystemOneError(
                f"Profile {profile.name!r}: model_kwargs.retry must be a mapping of "
                f"RetryPolicy fields, got {type(retry_config).__name__}"
            )
        # Defaulting to the SDK's own policy is the point of using it: it
        # honours Retry-After, jitters its backoff, and retries connection
        # and timeout errors, none of which a status-code loop of ours did.
        retry = RetryPolicy(**retry_config) if retry_config else None

        questions = questions_from_schema(output_schema)

        client_kwargs: dict[str, Any] = {}
        if profile.api_key is not None:
            client_kwargs["api_key"] = profile.api_key.get_secret_value()
        if base_url is not None:
            client_kwargs["base_url"] = str(base_url)
        if timeout is not None:
            client_kwargs["timeout"] = float(timeout)
        if retry is not None:
            client_kwargs["retry"] = retry
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        started = time.monotonic()
        try:
            with TypeSafeClient(**client_kwargs) as client:
                response = client.system_one(
                    state=prompt,
                    questions=_to_sdk_questions(questions),
                    model=profile.model,
                    extra_body=options or None,
                )
        except Exception as exc:
            # The SDK raises a typed hierarchy under TypeSafeError; wrapping
            # keeps consumers catching one chumak-shaped error per handler
            # while the original stays on __cause__.
            raise SystemOneError(
                f"System One call failed for profile {profile.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        duration_ms = (time.monotonic() - started) * 1000

        missing = set(questions) - set(response.answers)
        if missing:
            raise SystemOneError(f"System One omitted answers for: {sorted(missing)}")

        values = {
            name: _value_from_answer(
                name,
                response.answers[name],
                _unwrap_optional(output_schema.model_fields[name].annotation),
                noul_threshold=noul_threshold,
            )
            for name in questions
        }

        try:
            validated = output_schema.model_validate(values)
        except Exception as exc:
            raise SystemOneError(
                f"System One answers failed schema validation: {exc}; values={values!r}"
            ) from exc

        raw = SystemOneRaw(
            # The response names the resolved version (e.g. `jev-1.13.0`)
            # where the request named an alias (`jev-latest`). Prefer it, or
            # a silent vendor model bump is invisible in the ledger.
            model=str(response.model or profile.model),
            # The question map, not the whole request: the API key lives in
            # client headers and `raw` ends up in provenance records.
            questions=questions,
            answers={k: v.model_dump() for k, v in response.answers.items()},
            usage=response.usage.model_dump() if response.usage else {},
            request_id=_request_id_or_none(response),
            duration_ms=duration_ms,
        )
        # `state` is what actually reached the model, so it is what the meta
        # builder should hash for `prompt_actual_sha256`.
        return HandlerResult(payload=validated, raw=raw, rendered_prompt=prompt)
