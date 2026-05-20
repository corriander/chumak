"""Profile: a named inference configuration.

A flat shape with subprocess-only fields gated behind an explicit `handler`
discriminator. Per-handler subclassing is intentionally avoided while the
surface stays small — the validator below enforces the mutually-exclusive
field sets.

Profiles are user-authored. The library ships at most one generic example
profile; everything else is the consumer's. See `chumak.loader` for the
TOML / env-overlay machinery that builds these.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chumak.handlers.types import HandlerType, PromptDelivery


class Profile(BaseModel):
    """Named inference configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    handler: HandlerType
    model: str = Field(
        description=(
            "Model identifier. Authoritative for `langchain` profiles "
            "(passed to `init_chat_model`); descriptive metadata only for "
            "`subprocess` profiles (the command is what actually runs)."
        ),
    )
    description: str = ""
    prompt_version: str | None = Field(
        default=None,
        description=(
            "Optional human-readable prompt version label. Pair with "
            "`Provenance.prompt_template_sha256` if you care about drift."
        ),
    )
    temperature: float | None = None
    max_tokens: int | None = None
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Vendor-specific kwargs forwarded to the underlying SDK. "
            "For LangChain profiles, merged into `init_chat_model(...)` and "
            "takes precedence over top-level `temperature` / `max_tokens` on "
            "conflict. For subprocess profiles, must be empty (the command "
            "string is authoritative)."
        ),
    )

    command: str | None = Field(
        default=None,
        description="Subprocess profiles only. Argv parsed via `shlex.split`.",
    )
    prompt_delivery: PromptDelivery | None = Field(
        default=None,
        description="Subprocess profiles only. How the prompt reaches the CLI.",
    )
    timeout: float | None = Field(
        default=None,
        description="Subprocess profiles only. Seconds before the CLI is killed.",
    )

    @model_validator(mode="after")
    def _check_handler_fields(self) -> Profile:
        if self.handler is HandlerType.SUBPROCESS:
            if not self.command:
                raise ValueError(f"Profile {self.name!r}: subprocess profiles must set `command`")
            if self.prompt_delivery is None:
                raise ValueError(
                    f"Profile {self.name!r}: subprocess profiles must set `prompt_delivery`"
                )
            if self.model_kwargs:
                raise ValueError(
                    f"Profile {self.name!r}: subprocess profiles cannot set "
                    "`model_kwargs` (the command is authoritative)"
                )
        else:
            forbidden = {
                "command": self.command,
                "prompt_delivery": self.prompt_delivery,
                "timeout": self.timeout,
            }
            extras = [k for k, v in forbidden.items() if v is not None]
            if extras:
                raise ValueError(
                    f"Profile {self.name!r}: {extras} are only valid when "
                    f"handler == {HandlerType.SUBPROCESS.value!r}"
                )
        return self

    @property
    def is_subprocess(self) -> bool:
        return self.handler is HandlerType.SUBPROCESS
