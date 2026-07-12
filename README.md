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

Two built-in handlers:

- **`langchain`** — uses `langchain.chat_models.init_chat_model(profile.model)` so a
  single identifier (`anthropic:claude-opus-4-7`, `openai:gpt-5`, …) routes to the
  right provider. Structured output, citations, and token usage all handled. With an
  `output_schema` the call returns a validated instance; without one it returns the
  model's plain response text (a liveness/smoke probe or one-shot free-text question).
- **`subprocess`** — shells out to a CLI (`claude --print`, `codex exec`, etc.).
  Useful for prompt iteration via an existing, authorised tool.
  Schema is injected into the prompt as JSON Schema; stdout is parsed and validated.
  Requires an `output_schema` — untyped generation is a langchain-handler capability.

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

[model_kwargs]
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

# nested into model_kwargs
export MYAPP_VISION_PROFILE_CLAUDE_ACCOUNT_B_MODEL_KWARGS__API_KEY=sk-ant-...
```

This means a profile file can be effectively empty on disk (just declaring the
profile's existence and maybe an `extends`), with all values supplied by the
environment. You decide which fields are sensitive and never touch disk.

## Usage

```python
from pathlib import Path
from chumak import ProfileLoader, infer

loader = ProfileLoader(
    search_paths=[Path.home() / ".config/my-app/chumak/profiles"],
    env_prefix="MYAPP",
)
loader.names()                # -> ["claude", "claude-creative", ...]
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
result.payload          # -> MissionTitle(title=..., bounty=...)
result.citations        # -> [Citation, ...] (if the model supplied any)
result.meta             # -> Meta with cost, generated_at, model identity
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
result.meta.artefact_type    # "mission_title@v1"
result.meta.derived_from     # [...]
```

## Design notes

- **No domain knowledge**: chumak carries no built-in prompts, no role concepts
  (tactical/narrator etc. — that's an app concern; just name your profile).
- **LangChain is a handler, not the spine**: subprocess CLIs are first-class.
- **Provenance is opt-in**: omit `provenance=` and `meta.artefact_type` is `None`.
- **The lib never reads env directly** for its own settings. The env overlay
  for profiles is a deliberate, scoped exception, gated on the prefix the
  consumer passes in.

## Tooling

uv, Python 3.12+, ruff, ty, pytest. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the integration test, and quality-check commands.
