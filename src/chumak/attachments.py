"""Attachments: image inputs for `infer()`.

`infer(prompt=...)` takes a verbatim string. Vision-capable models also need
the image, and every API provider shapes that differently — exactly the
kind of plumbing a substrate hides. An `Attachment` names a local image
file; the langchain handler turns it into a multimodal message part, and
its digest is stamped into `Meta.produced_by.attachments` so the call stays
auditable without storing the bytes.

v1 scope: filesystem paths, image MIME types only. The subprocess handler
does not accept attachments — agent CLIs already have their own
path-reference idiom (``@{path}`` in the prompt) and merging the two
conventions silently would be confusing. The systemone handler does not
either: its input is a text state.

Leaf module — depends only on pydantic and the standard library.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath, ValidationInfo, field_validator


class AttachmentDigest(BaseModel):
    """What was attached, without the bytes. Lands in `ProducedBy.attachments`."""

    sha256: str
    mime: str


def digest_bytes(data: bytes, *, mime: str) -> AttachmentDigest:
    return AttachmentDigest(sha256=hashlib.sha256(data).hexdigest(), mime=mime)


class Attachment(BaseModel):
    """A local image file to send alongside the prompt.

    `mime` is sniffed from the file extension when omitted. Only ``image/*``
    types are accepted; anything else is rejected at validation, before any
    handler sees it. Frozen, so the validated MIME cannot be swapped for a
    non-image one after construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: FilePath = Field(description="Local image file. Must exist at construction time.")
    mime: str | None = Field(
        default=None,
        description="MIME type. Sniffed from the file extension when omitted.",
        validate_default=True,
    )

    @field_validator("mime")
    @classmethod
    def _resolve_and_check_mime(cls, mime: str | None, info: ValidationInfo) -> str | None:
        path = info.data.get("path")
        if path is None:  # `path` failed validation; its error is the one to report
            return mime
        if mime is None:
            mime, _ = mimetypes.guess_type(path.name)
            if mime is None:
                raise ValueError(
                    f"Attachment {str(path)!r}: cannot sniff MIME type from the file "
                    "extension; pass `mime=` explicitly"
                )
        if not mime.startswith("image/"):
            raise ValueError(
                f"Attachment {str(path)!r}: only image/* attachments are supported, got {mime!r}"
            )
        return mime

    @property
    def media_type(self) -> str:
        """The validated MIME type. Always set once the model has validated."""
        assert self.mime is not None
        return self.mime

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def digest(self) -> AttachmentDigest:
        return digest_bytes(self.read_bytes(), mime=self.media_type)
