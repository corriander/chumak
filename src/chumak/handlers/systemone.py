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
usage, which the meta builder mines via the `UsageSource` protocol.

Untyped calls are rejected: without a schema there are no questions to ask,
so there is nothing for Jev to answer. This mirrors the subprocess handler.

Wire format: https://docs.typesafe.ai/api.md (snapshot 2026-09-21).
"""

from __future__ import annotations

import enum
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from chumak.handlers.base import HandlerResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chumak.profile import Profile

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

# Price snapshot. Output tokens are free at this tier. Dated deliberately:
# this is a vendor price, and a stale constant that silently under-reports
# spend is worse than no constant at all. Cross-check against the Typesafe
# usage page when reconciling (ALS-96).
PRICE_SNAPSHOT_DATE = "2026-09-19"
INPUT_USD_PER_MTOK = 0.042
OUTPUT_USD_PER_MTOK = 0.0

# Per the API reference: Choice accepts at most 255 options, Score 2-10 levels.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Retried with backoff; every other status is raised immediately.
RETRY_STATUSES = frozenset({429, 529})

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 0.5
# A noul answer is a probability, not a boolean. Collapsing it to the
# schema's `bool` needs a cut point; 0.5 is the neutral one. Callers who
# care should read `raw` for the probability rather than move this.
_DEFAULT_NOUL_THRESHOLD = 0.5


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

    endpoint: str
    model: str
    request: dict[str, Any]
    response: dict[str, Any]
    answers: dict[str, Any] = dc_field(default_factory=dict)
    usage: dict[str, Any] = dc_field(default_factory=dict)
    duration_ms: float | None = None
    attempts: int = 1

    def token_usage(self) -> tuple[int | None, int | None]:
        """Satisfy `chumak.handlers.base.UsageSource` for the meta builder."""
        tokens_in = self.usage.get("input_tokens")
        tokens_out = self.usage.get("output_tokens")
        return (
            tokens_in if isinstance(tokens_in, int) else None,
            tokens_out if isinstance(tokens_out, int) else None,
        )

    def estimated_usd(self) -> float | None:
        """Best-effort USD for this call, at the dated price constants above.

        `None` when the response carried no usage block. Note this is an
        *estimate* against a snapshotted price — Typesafe's usage page is
        the billing source of truth.
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
        — for those the probability in `noul` is the whole signal.
        """
        answer = self.answers.get(question_id)
        if not isinstance(answer, dict):
            return None
        value = answer.get("confidence")
        return float(value) if isinstance(value, int | float) else None


class Transport(Protocol):
    """The HTTP seam. Swapped for a fake in unit tests."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> tuple[int, bytes]: ...


class UrllibTransport:
    """Default transport: stdlib only, so the handler adds no dependency.

    A single JSON POST does not justify pulling `httpx` or `typesafe-sdk`
    into a library whose whole point is being thin. Non-2xx responses are
    returned as `(status, body)` rather than raised, so the retry policy
    lives in one place (the handler) instead of being split across the
    transport boundary.
    """

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()


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
    """Translate a Pydantic schema into a Jev `questions` map.

    Public because the question map is the reusable artefact of an
    experiment: being able to inspect (and diff) exactly what was asked
    matters more than the handler's own plumbing. Question ids are field
    names, which is also how answers are keyed on the way back.
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


def _value_from_answer(
    name: str,
    answer: Any,
    annotation: Any,
    *,
    noul_threshold: float,
) -> Any:
    """Collapse one typed answer to the plain value the schema field wants."""
    if not isinstance(answer, dict):
        raise SystemOneError(f"Answer for {name!r} is not an object: {answer!r}")

    kind = answer.get("type")
    if kind == "noul":
        probability = answer.get("noul")
        if not isinstance(probability, int | float):
            raise SystemOneError(f"Answer for {name!r} has no numeric `noul`: {answer!r}")
        return bool(probability >= noul_threshold)
    if kind == "choice":
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise SystemOneError(f"Answer for {name!r} has no `choice` string: {answer!r}")
        return choice
    if kind == "score":
        score = answer.get("score")
        if not isinstance(score, int | float):
            raise SystemOneError(f"Answer for {name!r} has no numeric `score`: {answer!r}")
        # Score is probability-weighted and lands *between* levels, so an
        # int-annotated field has to be rounded rather than truncated.
        return round(score) if annotation is int else float(score)
    raise SystemOneError(f"Answer for {name!r} has unknown type {kind!r}")


class SystemOneHandler:
    """Experimental handler for TypeSafe System One (Jev).

    Transport-injectable so unit tests never touch the network. Retries are
    the handler's own responsibility (the vendor SDK would do it for us, but
    we deliberately do not depend on the SDK).

    Config rides `profile.model_kwargs` rather than new `Profile` fields —
    the same route `test_langchain_live.py` already uses for `base_url` and
    `api_key`, and the route the loader's env overlay reaches via
    `{APP}_PROFILE_{NAME}_MODEL_KWARGS__API_KEY`:

      - `api_key`        (required; never logged or echoed into `raw`)
      - `base_url`       (default: the public endpoint)
      - `timeout`        (seconds, default 30)
      - `max_retries`    (default 3, on 429/529 only)
      - `backoff_base`   (seconds, default 0.5; doubles per attempt)
      - `noul_threshold` (default 0.5)
    """

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self._transport = transport or UrllibTransport()
        self._sleep = sleep

    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
    ) -> HandlerResult:
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
        api_key = options.pop("api_key", None)
        if not api_key:
            raise SystemOneError(
                f"Profile {profile.name!r}: model_kwargs.api_key is required. Supply it via "
                f"the env overlay ({{APP}}_PROFILE_{profile.name.replace('-', '_').upper()}"
                "_MODEL_KWARGS__API_KEY) rather than committing it to TOML."
            )
        endpoint = str(options.pop("base_url", DEFAULT_ENDPOINT))
        timeout = float(options.pop("timeout", _DEFAULT_TIMEOUT))
        max_retries = int(options.pop("max_retries", _DEFAULT_MAX_RETRIES))
        backoff_base = float(options.pop("backoff_base", _DEFAULT_BACKOFF_BASE))
        noul_threshold = float(options.pop("noul_threshold", _DEFAULT_NOUL_THRESHOLD))

        questions = questions_from_schema(output_schema)
        body: dict[str, Any] = {
            "state": prompt,
            "model": profile.model,
            "questions": questions,
        }
        # Anything left over is forwarded verbatim, so a new top-level API
        # field does not require a chumak release to reach.
        body.update(options)

        payload_bytes = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        started = time.monotonic()
        status, response_bytes, attempts = self._post_with_retry(
            endpoint,
            headers=headers,
            body=payload_bytes,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        duration_ms = (time.monotonic() - started) * 1000

        if status != 200:
            raise SystemOneError(
                f"System One returned HTTP {status} for profile {profile.name!r}: "
                f"{_describe_error(response_bytes)}"
            )

        try:
            response = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise SystemOneError(
                f"System One returned non-JSON: {exc}; "
                f"body (first 500 bytes): {response_bytes[:500]!r}"
            ) from exc
        if not isinstance(response, dict):
            raise SystemOneError(
                f"System One returned a non-object body: {type(response).__name__}"
            )

        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise SystemOneError(f"System One response has no `answers` map: {response!r}")

        missing = set(questions) - set(answers)
        if missing:
            raise SystemOneError(f"System One omitted answers for: {sorted(missing)}")

        values = {
            name: _value_from_answer(
                name,
                answers[name],
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

        usage = response.get("usage")
        raw = SystemOneRaw(
            endpoint=endpoint,
            # The response names the resolved version (e.g. `jev-1.13.0`)
            # where the request named an alias (`jev-latest`). Prefer it.
            model=str(response.get("model") or profile.model),
            # The API key lives in the headers, which are not stored here —
            # `raw` ends up in provenance records and must stay clean.
            request=body,
            response=response,
            answers=answers,
            usage=usage if isinstance(usage, dict) else {},
            duration_ms=duration_ms,
            attempts=attempts,
        )
        # `state` is what actually reached the model, so it is what the meta
        # builder should hash for `prompt_actual_sha256`.
        return HandlerResult(payload=validated, raw=raw, rendered_prompt=prompt)

    def _post_with_retry(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
        max_retries: int,
        backoff_base: float,
    ) -> tuple[int, bytes, int]:
        """POST, retrying 429/529 with exponential backoff.

        Returns the last `(status, body, attempts)` even when every attempt
        was rate-limited — the caller turns a non-200 into the error, so the
        response body survives to explain itself.
        """
        status, response_bytes = 0, b""
        attempts = 0
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            status, response_bytes = self._transport.post(
                endpoint, headers=headers, body=body, timeout=timeout
            )
            if status not in RETRY_STATUSES or attempt == max_retries:
                return status, response_bytes, attempts
            self._sleep(backoff_base * (2**attempt))
        return status, response_bytes, attempts


def _describe_error(response_bytes: bytes) -> str:
    """Best-effort rendering of an error body, truncated for log safety."""
    try:
        parsed = json.loads(response_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return repr(response_bytes[:300])
    return json.dumps(parsed)[:300]
