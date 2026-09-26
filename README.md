# chumak

A thin **inference substrate** for Python projects: user-authored profiles, LangChain
as a handler, optional provenance/meta on every response.

> *Chumaks* (Чумаки) were wandering Ukrainian salt-traders who traversed the steppe
> between distant places. They named the Milky Way after themselves —
> *Чумацький Шлях*, the Chumaks' Way — because they navigated by it.

## What it is

A small library that abstracts away which LLM you're calling and how.

1. Author **profiles** (TOML files) under the app's XDG config dir.
2. Load a profile via `ProfileLoader` (with inheritance + env-var overrides)
3. Call `infer(prompt=..., output_schema=..., profile=...)` and get back a validated
   pydantic payload, normalised citations, and (optionally) a provenance `Meta` stamp.
   `output_schema` is **optional**: omit it for an untyped call whose `payload` is the
   model's plain response text (langchain handler only — see below).

Three built-in handlers:

- **`langchain`** — uses `langchain.chat_models.init_chat_model(profile.model)` so a
  single identifier (`anthropic:claude-opus-4-7`, `openai:gpt-5`, …) routes to the
  right provider. Structured output, citations, and token usage all handled. With an
  `output_schema` the call returns a validated instance; without one it returns the
  model's plain response text (a liveness/smoke probe or one-shot free-text question).
- **`subprocess`** — shells out to a CLI (`claude --print`, `codex exec`, etc.).
  Useful for prompt iteration via an existing, authorised tool.
  Schema is injected into the prompt as JSON Schema; stdout is parsed and validated.
  Requires an `output_schema` — untyped generation is a langchain-handler capability.
- **`systemone`** — ⚠️ **experimental.** Calls TypeSafe's System One decision model
  (Jev). Unlike the other two this is not a text generator: it answers *typed
  questions* about a state and returns calibrated probabilities. See below.

### The `systemone` handler (experimental)

[TypeSafe System One](https://docs.typesafe.ai/concepts/system-one) takes a `state`
plus a map of typed questions and answers all of them in one non-autoregressive pass.
It emits no text.

chumak bridges its contract by **deriving the question map from `output_schema`**, so
consumers call `infer()` exactly as they would for any other handler:

| Field type       | Question | Criteria                                           |
| ---------------- | -------- | -------------------------------------------------- |
| `bool`           | `noul`   | optional `{"true": ..., "false": ...}`             |
| `Literal` / Enum | `choice` | option → rubric map (max 255 options)              |
| `int` / `float`  | `score`  | **required** ordered list of 2–10 level rubrics    |

A field's `description` becomes the question; criteria come from
`Field(json_schema_extra={"criteria": ...})`.

```python
class Triage(BaseModel):
    department: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle this?",
        json_schema_extra={
            "criteria": {"billing": "Payments…", "technical": "Bugs…", "sales": "Pricing…"}
        },
    )
    is_urgent: bool = Field(description="Does this convey urgency?")
    frustration: int = Field(
        description="How frustrated is the customer?",
        json_schema_extra={"criteria": ["Calm", "Frustrated", "Very angry"]},
    )


result = chumak.infer(prompt=ticket_text, output_schema=Triage, profile=jev)
result.payload.department  # "billing"
result.meta.cost.tokens_in  # 296
```

The payload is a plain validated schema instance. **Probabilities and `confidence`
are not on it** — they ride on the handler's `raw` (a `SystemOneRaw`), along with
token usage and call duration. Note one asymmetry in the vendor API: choice and score
answers carry `confidence`, **noul answers do not** — there the probability itself is
the whole signal.

Untyped calls are rejected: no schema means no questions, and Jev generates no text.

Install the optional extra: `uv sync --extra typesafe` (or `pip install 'chumak[typesafe]'`).

**Retries are the vendor SDK's job, not ours.** Its `RetryPolicy` honours the
`Retry-After` header, jitters its backoff, and retries connection and timeout errors —
none of which a hand-rolled status-code loop does, and all of which matter when a
consumer walks a corpus in a tight loop. Override it per profile via `model_kwargs.retry`.

```toml
# ~/.config/<your-app>/chumak/profiles/jev.toml
handler = "systemone"
model = "jev-latest"
# No key in this file: the SDK reads TYPESAFE_API_KEY, or set a per-profile one
# via {APP}_PROFILE_JEV_API_KEY. See "API keys" below.

[model_kwargs]
timeout = 30

[model_kwargs.retry]      # splatted into the SDK's RetryPolicy
max_retries = 3
backoff_max = 10.0
```

`result.meta.cost` carries `tokens_in`, `tokens_out` **and `usd`** — the handler prices
its own calls from dated constants, so the library never holds a table of every model's
pricing. `raw.request_id` is the handle for reconciling a call against TypeSafe's own
usage page.

## Profiles

Profiles are user-authored. chumak ships at most one generic example
(`anthropic-claude-opus-4-7` via the LangChain handler). Everything else is yours.

Profiles live in consumer app directory, e.g. `~/.config/<your-app>/chumak/profiles/`.
chumak does not impose a config dir; the app passes `search_paths` to `ProfileLoader`.

### File shape

```toml
# ~/.config/galops-vision/chumak/profiles/claude.toml
handler = "langchain"
model = "anthropic:claude-opus-4-7"
temperature = 0.0

[model_kwargs]
max_tokens = 4096
```

### Inheritance

```toml
# claude-account-b.toml
extends = "claude"
# api_key sourced from env — see below
```

### Env-var overlay

Every field on a profile is overridable from the environment. chumak does not
provision special fields; the convention is uniform:

```
{APP_PREFIX}_PROFILE_{PROFILE_NAME}_{FIELD_PATH}
```

with `__` as the nested-field delimiter (single `_` stays inside field names):

```sh
# top-level field
export MYAPP_VISION_PROFILE_CLAUDE_MODEL=anthropic:claude-haiku-4-5

# the API key is a top-level field too
export MYAPP_VISION_PROFILE_CLAUDE_ACCOUNT_B_API_KEY=sk-ant-...

# nested into model_kwargs
export MYAPP_VISION_PROFILE_CLAUDE_MODEL_KWARGS__MAX_TOKENS=8192
```

This means a profile file can be effectively empty on disk (just declaring the
profile's existence and maybe an `extends`), with all values supplied by the
environment. You decide which fields are sensitive and never touch disk.

### API keys

A profile's credential is its top-level `api_key` field, held as a pydantic
`SecretStr`: wherever the profile is rendered (`repr`, logs, dumps, a traceback
capturing it) the key shows as `**********`. Handlers unwrap it only when they build
the SDK client, so a failure inside that SDK call can still hold the plain value in
its own frames.

- **Leave it unset** and the SDK reads its own standard variable
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`, …). This is the simplest
  route, and the one secret managers such as varlock inject into.
- **Set it** when one provider needs a different key per profile (two accounts, say),
  via the env overlay rather than TOML.
- **`model_kwargs.api_key` is rejected.** `model_kwargs` is not treated as secret, so
  keep credentials out of it, custom auth headers included.
- **Watch custom endpoints.** An `openai:` profile with its own `base_url` (llama.cpp,
  LiteLLM, any OpenAI-compatible proxy) and no `api_key` falls back to `OPENAI_API_KEY`,
  and LangChain sends that key to the custom endpoint. Give such profiles their own key;
  for a server without auth, a placeholder such as `api_key = "sk-no-auth"` in TOML is
  fine, since it is not a secret.

When you build a profile in code, pass a `SecretStr`. Pydantic accepts a plain string
at runtime, but some type checkers hold the argument to the declared type,
`SecretStr | None`, and reject it:

```python
from pydantic import SecretStr

from chumak import HandlerType, Profile

profile = Profile(
    name="local-edge",
    handler=HandlerType.LANGCHAIN,
    model="openai:mistral-7b",
    api_key=SecretStr("sk-no-auth"),  # placeholder for an unauthenticated server
    model_kwargs={"base_url": "http://localhost:8080/v1"},
)
```

Profile validation errors never echo input values, so a key routed to the wrong
field doesn't end up in the message.

## Usage

```python
from pathlib import Path
from chumak import ProfileLoader, infer

loader = ProfileLoader(
    search_paths=[Path.home() / ".config/my-app/chumak/profiles"],
    env_prefix="MYAPP",
)
loader.names()  # -> ["claude", "claude-creative", ...]
profile = loader.load("claude")

from pydantic import BaseModel


class AnneSchema(BaseModel):
    title: str
    value: int


result = infer(
    prompt="Extract title and value from this text: ...",
    output_schema=AnneSchema,
    profile=profile,
)
result.payload  # -> MissionTitle(title=..., bounty=...)
result.citations  # -> [Citation, ...] (if the model supplied any)
result.meta  # -> Meta with cost, generated_at, model identity
```

### With provenance

```python
from chumak import Provenance

result = infer(
    prompt="...",
    output_schema=AnneSchema,
    profile=profile,
    provenance=Provenance(
        artefact_type="model@v1",
        artefact_id="artifact-type:2026-05-20T12:34:56Z",
    ),
)
result.meta.artefact_type  # "mission_title@v1"
result.meta.derived_from  # [...]
```

### Orchestrate your own loop

`infer()` is one-shot by design. If you run the loop yourself — an agent graph, a
multi-turn conversation, a streaming UI — chumak still answers the two questions it
exists for: *which model, configured how* and *what did this call produce*. You own
everything in between.

```python
from chumak import build_meta, resolve_model

model = resolve_model(profile)  # the same chat model infer() would use for this profile
response = model.invoke(prompt)  # or .stream(), or hand `model` to your graph
meta = build_meta(response, profile=profile, prompt=prompt)  # same envelope infer() stamps
```

- `resolve_model` is a **langchain-handler capability**. A subprocess profile pins its
  model inside a CLI command, and a `systemone` profile's model is a decision engine
  rather than a chat model; neither has anything to hand out, so both raise
  `ProfileCapabilityError` rather than pretending.
- `build_meta` stamps one call. Compose run-level lineage yourself with
  `Provenance` / `ArtefactRef` / `derived_from` — chumak records, it never runs the loop.
- No graph, conversation, callback, or streaming abstractions live here; LangChain /
  LangGraph objects are the consumer's domain.

## Design notes

- **No domain knowledge**: chumak carries no built-in prompts, no role concepts
  (tactical/narrator etc. — that's an app concern; just name your profile).
- **LangChain is a handler, not the spine**: subprocess CLIs are first-class.
- **Provenance is opt-in**: omit `provenance=` and `meta.artefact_type` is `None`.
- **Meta is safe to persist**: `meta.produced_by` records the profile name, model and
  prompt hashes, never `model_kwargs` or the key.
- **The lib never reads env directly** for its own settings. The env overlay
  for profiles is a deliberate, scoped exception, gated on the prefix the
  consumer passes in.

## Tooling

uv, Python 3.12+, ruff, ty, pytest. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the integration test, and quality-check commands.
