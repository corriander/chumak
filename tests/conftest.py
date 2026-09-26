"""Shared fixtures.

Heavy use of fixtures over hand-rolled setup, matching the galops / space-hulk
house style. Two themes:

  - **Profile factories** for the loader / profile tests: build TOML files
    in a temp dir and hand back a `ProfileLoader` pointed at it.
  - **Fake handlers** for surface / meta tests: skip real LangChain /
    subprocess transports and feed a known `HandlerResult` into the
    coordinator.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from chumak.attachments import Attachment
from chumak.handlers import HANDLER_REGISTRY
from chumak.handlers.base import HandlerResult
from chumak.handlers.types import HandlerType
from chumak.loader import ProfileLoader
from chumak.profile import Profile


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    """A fresh per-test directory for profile TOML files."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def write_profile(profile_dir: Path) -> Callable[[str, str], Path]:
    """Write a TOML profile file and return its path."""

    def _write(name: str, body: str) -> Path:
        path = profile_dir / f"{name}.toml"
        path.write_text(body)
        return path

    return _write


@pytest.fixture
def make_loader(profile_dir: Path) -> Callable[..., ProfileLoader]:
    """Build a `ProfileLoader` pointed at the per-test profile dir.

    Pass `env=` to inject a synthetic environment without touching `os.environ`.
    """

    def _make(
        *,
        env_prefix: str = "TESTAPP",
        env: Mapping[str, str] | None = None,
    ) -> ProfileLoader:
        return ProfileLoader(
            search_paths=[profile_dir],
            env_prefix=env_prefix,
            env=env or {},
        )

    return _make


class _FakeHandler:
    """A handler that returns whatever `HandlerResult` it was constructed with.

    Registered into `HANDLER_REGISTRY` by the `stub_handler` fixture below
    via a class-level slot, so the no-arg constructor LangChain et al. use
    still works.
    """

    _result: HandlerResult | None = None

    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
        attachments: Sequence[Attachment] = (),
    ) -> HandlerResult:
        assert _FakeHandler._result is not None, (
            "stub_handler fixture must be set up before execute()"
        )
        # Echo back the supplied payload class so surface tests get a real
        # instance of their `output_schema`.
        return _FakeHandler._result


@pytest.fixture
def stub_handler() -> Iterator[Callable[[Any], None]]:
    """Replace the LANGCHAIN handler in `HANDLER_REGISTRY` with a fake whose
    `HandlerResult` the test controls. Restored on teardown.
    """
    original = HANDLER_REGISTRY[HandlerType.LANGCHAIN]
    HANDLER_REGISTRY[HandlerType.LANGCHAIN] = _FakeHandler

    def _set(result: HandlerResult) -> None:
        _FakeHandler._result = result

    try:
        yield _set
    finally:
        HANDLER_REGISTRY[HandlerType.LANGCHAIN] = original
        _FakeHandler._result = None


class SampleOutput(BaseModel):
    """A small output schema for tests that need one."""

    title: str
    bounty: int


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A real, minimal PNG of one flat colour — no imaging library needed.

    Valid enough for any decoder (and any vision model): signature, IHDR,
    one zlib-compressed IDAT of filter-0 scanlines, IEND.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def red_png(tmp_path: Path) -> Path:
    """A 16x16 solid-red PNG on disk."""
    path = tmp_path / "red.png"
    path.write_bytes(solid_png(16, 16, (255, 0, 0)))
    return path


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit a live inference backend.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="Pass --integration to run live backend tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
