"""`Attachment`: MIME sniffing, image-only gate, digest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from chumak.attachments import Attachment


def test_mime_sniffed_from_extension(red_png: Path) -> None:
    att = Attachment(path=red_png)
    assert att.mime == "image/png"
    assert att.media_type == "image/png"


def test_explicit_mime_is_honoured(tmp_path: Path) -> None:
    # Extension says nothing useful; the caller knows better.
    path = tmp_path / "capture.bin"
    path.write_bytes(b"\xff\xd8\xff")
    att = Attachment(path=path, mime="image/jpeg")
    assert att.media_type == "image/jpeg"


def test_unsniffable_extension_requires_explicit_mime(tmp_path: Path) -> None:
    path = tmp_path / "capture"  # no extension: nothing for `mimetypes` to go on
    path.write_bytes(b"x")
    with pytest.raises(ValidationError, match="cannot sniff MIME type"):
        Attachment(path=path)


def test_non_image_rejected_by_sniff(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    with pytest.raises(ValidationError, match="only image/\\* attachments"):
        Attachment(path=path)


def test_non_image_rejected_when_explicit(red_png: Path) -> None:
    with pytest.raises(ValidationError, match="only image/\\* attachments"):
        Attachment(path=red_png, mime="application/pdf")


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Attachment(path=tmp_path / "nope.png")


def test_digest_is_sha256_of_bytes(red_png: Path) -> None:
    digest = Attachment(path=red_png).digest()
    assert digest.sha256 == hashlib.sha256(red_png.read_bytes()).hexdigest()
    assert digest.mime == "image/png"
