"""Subprocess-backed handler.

Shells out to a CLI tool. CLI tools don't expose structured-output APIs, so
the schema is embedded in the prompt body — the model is instructed to emit
JSON matching the schema, and stdout is parsed and validated.

The profile shape stays minimal: a verbatim `command` (parsed via
`shlex.split`), a delivery channel (`stdin` or trailing argv), and a
timeout. Output is expected as JSON on stdout, optionally wrapped in a
fenced code block.

Attachments are opt-in per profile. CLIs that read images take them as a
file path written into the prompt, and each spells that reference its own
way, so the profile's `attachment_ref` template says how (`{path}` for
`ollama run`). With it set, the handler writes one reference per attachment
after the caller's prompt, each on its own line, in order, and ahead of the
schema instructions. Without it, attachments are rejected rather than
dropped. CLIs that take images as argv flags instead aren't covered.

A reference is not a send: chumak never sees what the CLI reads. So the
handler hashes each file as it renders the prompt, and reports the digests
as `attachments_referenced`, not `attachments`. A file it can't read fails
the call before anything is spawned.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from chumak.attachments import Attachment, AttachmentDigest
from chumak.errors import ProfileCapabilityError
from chumak.handlers.base import HandlerResult
from chumak.handlers.types import PromptDelivery

if TYPE_CHECKING:
    from chumak.profile import Profile

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class SubprocessRaw:
    command: str
    returncode: int
    stdout: str
    stderr: str


def _build_subprocess_prompt(
    prompt: str, output_schema: type[BaseModel], references: Sequence[str] = ()
) -> str:
    if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
        raise TypeError("output_schema must be a Pydantic BaseModel subclass")
    schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
    # No references, no change: a text-only call renders, and hashes, exactly
    # as it did before attachments existed.
    body = "\n\n".join([prompt, "\n".join(references)]) if references else prompt
    return (
        f"{body}\n\n"
        "Respond with a single JSON object matching the following JSON Schema. "
        "Do not include prose, code fences, or commentary outside the JSON.\n\n"
        f"```json\n{schema_json}\n```\n"
    )


def _reference_attachments(
    attachments: Sequence[Attachment], template: str
) -> tuple[list[str], list[AttachmentDigest]]:
    """Render one reference per attachment, and hash each file.

    Paths are made absolute, since the CLI's working directory is the
    caller's and may not be where the attachment was named from.
    """
    references: list[str] = []
    digests: list[AttachmentDigest] = []
    for attachment in attachments:
        path = str(Path(attachment.path).absolute())
        # Each reference gets a line of its own; a path that breaks the line
        # would split it into text the CLI doesn't read as a reference.
        if "\n" in path or "\r" in path:
            raise ValueError(
                f"Attachment path {path!r} contains a line break, so it can't be written "
                "into the prompt as a reference"
            )
        references.append(template.replace("{path}", path))
        digests.append(attachment.digest())
    return references, digests


def _extract_json(stdout: str) -> str:
    """Bare JSON wins over fence extraction: a fence *inside* a JSON string
    value must never be mistaken for the payload's wrapper.
    """
    stripped = stdout.strip()
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        match = _FENCE_RE.search(stripped)
        return match.group(1) if match else stripped
    return stripped


class SubprocessHandler:
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None,
        profile: Profile,
        attachments: Sequence[Attachment] = (),
    ) -> HandlerResult:
        if not profile.is_subprocess:
            raise ProfileCapabilityError(
                f"SubprocessHandler called with non-subprocess profile {profile.name!r}"
            )
        if attachments and profile.attachment_ref is None:
            raise ProfileCapabilityError(
                f"Subprocess profile {profile.name!r} does not accept attachments "
                f"({len(attachments)} given): set `attachment_ref` to the template its CLI "
                "uses to reference a file in the prompt (e.g. `{path}` for `ollama run`)"
            )
        if output_schema is None:
            # The subprocess contract *is* the injected JSON Schema: without one
            # there is nothing to instruct the CLI to emit or to validate its
            # stdout against. Untyped generation is a LangChain-handler
            # capability; a subprocess profile must supply a schema.
            raise ValueError(
                f"SubprocessHandler requires an output_schema (profile {profile.name!r}): "
                "untyped/plain-text inference is only supported by the langchain handler"
            )
        assert profile.command is not None
        assert profile.prompt_delivery is not None

        references: list[str] = []
        digests: list[AttachmentDigest] = []
        if attachments:
            assert profile.attachment_ref is not None
            references, digests = _reference_attachments(attachments, profile.attachment_ref)
        full_prompt = _build_subprocess_prompt(prompt, output_schema, references)
        argv = shlex.split(profile.command)
        proc = self._run(argv, full_prompt, profile.prompt_delivery, profile.timeout)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess {profile.command!r} exited {proc.returncode}: "
                f"stderr={proc.stderr.strip()!r}"
            )

        json_text = _extract_json(proc.stdout)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Subprocess {profile.command!r} returned non-JSON output: {e}; "
                f"stdout (first 500 chars): {proc.stdout[:500]!r}"
            ) from e

        try:
            payload = output_schema.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Subprocess output failed schema validation: {e}; "
                f"data (first 500 chars): {json.dumps(data)[:500]}"
            ) from e

        return HandlerResult(
            payload=payload,
            raw=SubprocessRaw(
                command=profile.command,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            ),
            rendered_prompt=full_prompt,
            attachments_referenced=digests,
        )

    def _run(
        self,
        argv: list[str],
        prompt: str,
        delivery: PromptDelivery,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        if delivery is PromptDelivery.STDIN:
            return subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        if delivery is PromptDelivery.ARG:
            return subprocess.run(
                [*argv, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        raise ValueError(f"Unknown prompt_delivery: {delivery}")
