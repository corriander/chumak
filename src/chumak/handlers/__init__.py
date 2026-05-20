"""Handlers package.

Owns the `HandlerType` discriminator and the `HANDLER_REGISTRY` mapping each
`HandlerType` to its concrete handler class. Adding a new handler type means
dropping a module here, extending `HandlerType` in `types.py`, and adding
the class to the registry below — no edits in `surface.py` or `profile.py`.
"""

from chumak.handlers.base import Handler, HandlerResult
from chumak.handlers.langchain import LangChainHandler
from chumak.handlers.subprocess import SubprocessHandler
from chumak.handlers.types import HandlerType, PromptDelivery

HANDLER_REGISTRY: dict[HandlerType, type[Handler]] = {
    HandlerType.LANGCHAIN: LangChainHandler,
    HandlerType.SUBPROCESS: SubprocessHandler,
}

__all__ = [
    "HANDLER_REGISTRY",
    "Handler",
    "HandlerResult",
    "HandlerType",
    "LangChainHandler",
    "PromptDelivery",
    "SubprocessHandler",
]
