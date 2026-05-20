"""ProfileLoader: discovery, inheritance, env-var overlay.

The loader's job is to take a TOML file on disk + an `extends` chain + the
process environment and produce a validated `Profile`. The tests below
exercise each layer independently and together.
"""

from __future__ import annotations

import textwrap

import pytest

from chumak.handlers.types import HandlerType, PromptDelivery
from chumak.loader import ProfileCycleError, ProfileNotFoundError


@pytest.fixture
def claude_base_body() -> str:
    return textwrap.dedent(
        """
        handler = "langchain"
        model = "anthropic:claude-opus-4-7"
        temperature = 0.0

        [model_kwargs]
        max_tokens = 4096
        """
    ).strip()


def test_load_minimal_profile(write_profile, make_loader, claude_base_body) -> None:
    write_profile("claude", claude_base_body)
    loader = make_loader()
    profile = loader.load("claude")

    assert profile.name == "claude"
    assert profile.handler is HandlerType.LANGCHAIN
    assert profile.model == "anthropic:claude-opus-4-7"
    assert profile.temperature == 0.0
    assert profile.model_kwargs == {"max_tokens": 4096}


def test_missing_profile_raises(make_loader) -> None:
    loader = make_loader()
    with pytest.raises(ProfileNotFoundError, match="not found"):
        loader.load("ghost")


def test_extends_merges_parent_then_child(write_profile, make_loader, claude_base_body) -> None:
    """Child overrides scalars; nested dicts merge key-by-key."""
    write_profile("claude", claude_base_body)
    write_profile(
        "claude-creative",
        textwrap.dedent(
            """
            extends = "claude"
            temperature = 0.7

            [model_kwargs]
            top_p = 0.95
            """
        ).strip(),
    )

    loader = make_loader()
    profile = loader.load("claude-creative")

    assert profile.model == "anthropic:claude-opus-4-7"
    assert profile.temperature == 0.7
    # Child's `top_p` joined parent's `max_tokens` — dict merged, not replaced.
    assert profile.model_kwargs == {"max_tokens": 4096, "top_p": 0.95}


def test_extends_cycle_detected(write_profile, make_loader) -> None:
    write_profile("a", 'extends = "b"\nhandler = "langchain"\nmodel = "x"\n')
    write_profile("b", 'extends = "a"\nhandler = "langchain"\nmodel = "x"\n')
    loader = make_loader()
    with pytest.raises(ProfileCycleError, match="cycle"):
        loader.load("a")


def test_env_overlay_overrides_scalar(write_profile, make_loader, claude_base_body) -> None:
    write_profile("claude", claude_base_body)
    loader = make_loader(env={"TESTAPP_PROFILE_CLAUDE_TEMPERATURE": "0.9"})
    profile = loader.load("claude")
    assert profile.temperature == 0.9


def test_env_overlay_into_nested_model_kwargs(
    write_profile, make_loader, claude_base_body
) -> None:
    """`__` is the nest delimiter; `_` lives inside field names like `model_kwargs`."""
    write_profile("claude", claude_base_body)
    loader = make_loader(
        env={
            "TESTAPP_PROFILE_CLAUDE_MODEL_KWARGS__API_KEY": "sk-from-env",
            "TESTAPP_PROFILE_CLAUDE_MODEL_KWARGS__MAX_TOKENS": "8192",
        }
    )
    profile = loader.load("claude")
    assert profile.model_kwargs["api_key"] == "sk-from-env"
    assert profile.model_kwargs["max_tokens"] == 8192


def test_env_overlay_into_hyphenated_profile_name(write_profile, make_loader) -> None:
    """A profile file `claude-account-b.toml` is reached via
    `..._PROFILE_CLAUDE_ACCOUNT_B_...`."""
    write_profile(
        "claude-account-b",
        textwrap.dedent(
            """
            extends = "_not_used_"
            handler = "langchain"
            model = "anthropic:claude-opus-4-7"
            """
        )
        .strip()
        .replace('extends = "_not_used_"\n', ""),
    )
    loader = make_loader(
        env={
            "TESTAPP_PROFILE_CLAUDE_ACCOUNT_B_MODEL_KWARGS__API_KEY": "sk-acct-b",
        }
    )
    profile = loader.load("claude-account-b")
    assert profile.model_kwargs == {"api_key": "sk-acct-b"}


def test_empty_file_can_be_fully_env_driven(write_profile, make_loader) -> None:
    """The point: a TOML file can be effectively empty, with env declaring
    the actual configuration. Useful when every value is sensitive.
    """
    write_profile("phantom", "")
    loader = make_loader(
        env={
            "TESTAPP_PROFILE_PHANTOM_HANDLER": "langchain",
            "TESTAPP_PROFILE_PHANTOM_MODEL": "anthropic:claude-opus-4-7",
            "TESTAPP_PROFILE_PHANTOM_MODEL_KWARGS__API_KEY": "sk-from-env",
        }
    )
    profile = loader.load("phantom")
    assert profile.handler is HandlerType.LANGCHAIN
    assert profile.model_kwargs == {"api_key": "sk-from-env"}


def test_env_overlay_disabled_when_prefix_empty(
    write_profile, make_loader, claude_base_body
) -> None:
    write_profile("claude", claude_base_body)
    loader = make_loader(
        env_prefix="",
        env={"TESTAPP_PROFILE_CLAUDE_TEMPERATURE": "0.9"},
    )
    profile = loader.load("claude")
    assert profile.temperature == 0.0  # untouched


def test_env_overlay_does_not_match_other_prefixes(
    write_profile, make_loader, claude_base_body
) -> None:
    write_profile("claude", claude_base_body)
    loader = make_loader(
        env_prefix="GALOPS_VISION",
        env={
            "OTHERAPP_PROFILE_CLAUDE_TEMPERATURE": "0.9",
            "GALOPS_VISION_PROFILE_CLAUDE_TEMPERATURE": "0.5",
        },
    )
    profile = loader.load("claude")
    assert profile.temperature == 0.5


def test_env_overlay_json_values(write_profile, make_loader) -> None:
    """JSON-looking env values are parsed (lists, bools, numbers, null)."""
    write_profile(
        "cli",
        textwrap.dedent(
            """
            handler = "subprocess"
            model = "claude-opus-4-7"
            command = "claude --print"
            prompt_delivery = "stdin"
            """
        ).strip(),
    )
    loader = make_loader(
        env={
            "TESTAPP_PROFILE_CLI_TIMEOUT": "120.5",
        }
    )
    profile = loader.load("cli")
    assert profile.timeout == 120.5
    assert profile.prompt_delivery is PromptDelivery.STDIN


def test_list_returns_discovered_profile_names(
    write_profile, make_loader, claude_base_body
) -> None:
    write_profile("claude", claude_base_body)
    write_profile("claude-creative", 'extends = "claude"\ntemperature = 0.7\n')
    loader = make_loader()
    assert sorted(loader.names()) == ["claude", "claude-creative"]


def test_load_all(write_profile, make_loader, claude_base_body) -> None:
    write_profile("claude", claude_base_body)
    write_profile(
        "claude-creative",
        textwrap.dedent(
            """
            extends = "claude"
            temperature = 0.7
            """
        ).strip(),
    )
    loader = make_loader()
    profiles = loader.load_all()
    assert set(profiles) == {"claude", "claude-creative"}
    assert profiles["claude-creative"].temperature == 0.7
