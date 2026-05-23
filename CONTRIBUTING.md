# Contributing

## Principles

- **Keep the substrate thin.** chumak coordinates handlers, profiles, and meta. It does not own prompts, domain concepts, or secret management — those are consumer concerns.
- **Handlers are pluggable.** Adding a transport (LangChain, subprocess, …) is dropping a module under `src/chumak/handlers/`, extending `HandlerType`, and registering it in `HANDLER_REGISTRY`. No edits in `surface.py` or `profile.py`.
- **The library never reads env vars unprompted.** The profile env-overlay is the one exception, gated on the prefix the consumer passes to `ProfileLoader`.
- **Prefer fixtures over hand-rolled setup.** See `tests/conftest.py` (`write_profile`, `make_loader`, `stub_handler`).

## Dev setup

```bash
git clone <repo>
cd chumak
uv sync                         # core deps + dev tools (ruff, ty, pytest, pytest-mock, langchain-anthropic)
uv sync --extra openai          # add when running the LangChain live integration test
```

Python 3.12+ (no 3.13-only syntax is used — keep it that way to stay broadly compatible with consumer projects).

## Tests

### Unit tests (default — fast, hermetic, no network)

```bash
uv run pytest
```

These cover every code path. The `stub_handler` fixture swaps the LangChain handler in `HANDLER_REGISTRY` for a deterministic fake so surface/meta tests never touch a real LLM. Use it when writing new tests that need a known `HandlerResult`.

Profile-loader tests use `write_profile` to drop TOML files into a per-test tmp dir and `make_loader` to build a `ProfileLoader` pointed at it. Pass `env=` to inject a synthetic environment without mutating `os.environ`.

### Integration test — LangChain handler against an OpenAI-compat backend

Marker-gated, off by default. Exercises chumak end-to-end through a real LangChain `init_chat_model` call against any OpenAI-API-compatible server: llama.cpp, vLLM, LiteLLM, real OpenAI, etc. This is the test that proves the `openai:<model>` + `base_url` wiring still works after handler / langchain version bumps.

```bash
uv sync --extra openai                                # install langchain-openai
uv run pytest --integration tests/test_langchain_live.py -v
```

Defaults: `http://localhost:8080/v1`, model `gpt-3.5-turbo`, dummy API key. Override per-run via env vars:

```bash
CHUMAK_TEST_OPENAI_URL=http://localhost:8000/v1 \
CHUMAK_TEST_OPENAI_MODEL=qwen2.5-7b-instruct \
  uv run pytest --integration tests/test_langchain_live.py -v
```

The test uses a tiny `ColourTag { colour: str, is_warm: bool }` schema — small enough that any reasonable 7B-class instruct model handles it. If you add coverage for a new handler or option, add a sibling test under the same marker and document the env vars it needs here.

#### Why no subprocess live test?

The subprocess handler is harder to gate (each CLI has its own auth / install requirements). Cover it with unit tests against a temporary script (`tests/test_subprocess_handler.py` does this), and rely on downstream consumers for true end-to-end exercise.

## Quality checks

Before opening a PR:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check src tests
```

All three must pass clean. CI will run the same.

## Adding a handler

1. Module under `src/chumak/handlers/<name>.py` exporting a class with `execute(prompt, output_schema, profile) -> HandlerResult`.
2. Add the discriminator to `HandlerType` in `src/chumak/handlers/types.py`.
3. Register the class in `HANDLER_REGISTRY` (`src/chumak/handlers/__init__.py`).
4. If the handler needs new profile fields, add them to `Profile` with the validator enforcing mutually-exclusive field sets per handler.
5. Add unit tests via `stub_handler` for the surface side and direct handler tests for transport-specific quirks.

## Adding a new profile field

Profile fields live in `src/chumak/profile.py`. The validator enforces which handler types may use which fields — extend it when you add fields that only make sense for a specific handler. The env-overlay walker in `loader.py` picks them up automatically as long as they're declared on the `Profile` model.
