"""Subprocess-backed handler.

Shells out to a CLI tool. CLI tools don't expose structured-output APIs, so
the schema is embedded in the prompt body — the model is instructed to emit
JSON matching the schema, and stdout is parsed and validated.

The profile shape stays minimal: a verbatim `command` (parsed via
`shlex.split`), a delivery channel (`stdin` or trailing argv), and a
timeout. Output is expected as JSON on stdout, optionally wrapped in a
fenced code block.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

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


def _build_subprocess_prompt(prompt: str, output_schema: type[BaseModel]) -> str:
    if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
        raise TypeError("output_schema must be a Pydantic BaseModel subclass")
    schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
    return (
        f"{prompt}\n\n"
        "Respond with a single JSON object matching the following JSON Schema. "
        "Do not include prose, code fences, or commentary outside the JSON.\n\n"
        f"```json\n{schema_json}\n```\n"
    )


def _extract_json(stdout: str) -> str:
    stripped = stdout.strip()
    match = _FENCE_RE.search(stripped)
    return match.group(1) if match else stripped


class SubprocessHandler:
    def execute(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        profile: Profile,
    ) -> HandlerResult:
        if not profile.is_subprocess:
            raise ValueError(
                f"SubprocessHandler called with non-subprocess profile {profile.name!r}"
            )
        assert profile.command is not None
        assert profile.prompt_delivery is not None

        full_prompt = _build_subprocess_prompt(prompt, output_schema)
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
