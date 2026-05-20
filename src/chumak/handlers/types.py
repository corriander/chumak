"""Handler-related enum types.

Leaf module — no dependencies on the rest of the package. Keeps the
discriminator/delivery enums importable from `profile.py` and each handler
module without import cycles.
"""

from __future__ import annotations

from enum import StrEnum


class HandlerType(StrEnum):
    LANGCHAIN = "langchain"
    SUBPROCESS = "subprocess"


class PromptDelivery(StrEnum):
    STDIN = "stdin"
    ARG = "arg"
