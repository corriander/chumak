"""chumak — thin inference substrate.

Public surface in `surface.infer`. Profile-loading in `loader.ProfileLoader`.
Response types in `response`. Handlers are pluggable via `handlers.HANDLER_REGISTRY`.

Consumers that run their own inference loop use the pair `resolve_model`
(which model, configured how) + `build_meta` (what did this call produce).

The library is subject-agnostic and carries no domain knowledge: no
built-in prompts, no role names, no per-domain artefact types. Consumers
build those on top.
"""

from chumak.attachments import Attachment, AttachmentDigest
from chumak.errors import ProfileCapabilityError
from chumak.handlers import HANDLER_REGISTRY, Handler, HandlerType, PromptDelivery
from chumak.handlers.langchain import resolve_model
from chumak.loader import (
    ProfileCycleError,
    ProfileLoader,
    ProfileLoaderError,
    ProfileNotFoundError,
)
from chumak.meta import build_meta
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
    "Attachment",
    "AttachmentDigest",
    "Citation",
    "Cost",
    "Handler",
    "HandlerType",
    "InferResult",
    "Meta",
    "ProducedBy",
    "Profile",
    "ProfileCapabilityError",
    "ProfileCycleError",
    "ProfileLoader",
    "ProfileLoaderError",
    "ProfileNotFoundError",
    "PromptDelivery",
    "Provenance",
    "build_meta",
    "infer",
    "resolve_model",
]
