"""chumak — thin inference substrate.

Public surface in `surface.infer`. Profile-loading in `loader.ProfileLoader`.
Response types in `response`. Handlers are pluggable via `handlers.HANDLER_REGISTRY`.

The library is subject-agnostic and carries no domain knowledge: no
built-in prompts, no role names, no per-domain artefact types. Consumers
build those on top.
"""

from chumak.handlers import HANDLER_REGISTRY, Handler, HandlerType, PromptDelivery
from chumak.loader import (
    ProfileCycleError,
    ProfileLoader,
    ProfileLoaderError,
    ProfileNotFoundError,
)
from chumak.profile import Profile
from chumak.response import (
    ArtefactRef,
    Citation,
    Cost,
    InferResult,
    Meta,
    ProducedBy,
    Provenance,
)
from chumak.surface import infer

__all__ = [
    "HANDLER_REGISTRY",
    "ArtefactRef",
    "Citation",
    "Cost",
    "Handler",
    "HandlerType",
    "InferResult",
    "Meta",
    "ProducedBy",
    "Profile",
    "ProfileCycleError",
    "ProfileLoader",
    "ProfileLoaderError",
    "ProfileNotFoundError",
    "PromptDelivery",
    "Provenance",
    "infer",
]
