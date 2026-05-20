"""Profile loading: TOML on disk + inheritance + env-var overlay.

The loader is the interesting bit. Three layers compose, in order:

  1. **TOML on disk.** A profile file `<name>.toml` lives somewhere under
     `search_paths`. Its raw contents are the starting dict.
  2. **`extends` chain.** If the file declares `extends = "<parent>"`, the
     parent profile is loaded (recursively) and the child is *deep-merged*
     on top: scalars override, dicts merge key-by-key, lists replace.
  3. **Environment overlay.** Every leaf field is overridable from the
     process environment. The env-var path is

         {APP_PREFIX}_PROFILE_{PROFILE_NAME_UPPER}_{FIELD_PATH_UPPER}

     with `__` as the nested-dict delimiter so single `_` can live inside
     field names (`model_kwargs`, `max_tokens`, ...) without ambiguity.

The result is validated as `Profile`. The library never reads env vars on
its own initiative — the overlay is gated on the prefix the consumer passes
to the `ProfileLoader` constructor.

This means a TOML file on disk can be effectively empty (just enough to
declare a profile's existence and maybe an `extends`), with all values —
including secrets — supplied at process start by env vars.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from chumak.profile import Profile

_NEST_DELIM = "__"


class ProfileLoaderError(Exception):
    """Anything went wrong loading a profile."""


class ProfileNotFoundError(ProfileLoaderError):
    """No TOML file matched a requested profile name across the search paths."""


class ProfileCycleError(ProfileLoaderError):
    """The `extends` chain forms a cycle."""


class ProfileLoader:
    """Resolves profile names to validated `Profile` instances.

    `search_paths` is a list of directories. The loader searches them in
    order for `<name>.toml`; the first match wins. This lets apps shadow
    bundled defaults with user overrides (user dir first, app default dir
    second).

    `env_prefix` is the app-specific prefix for the env overlay
    (e.g. `"GALOPS_VISION"`). Required — leaving it implicit would risk
    polluting unrelated apps' env namespaces. Pass an empty string to
    disable env overlay entirely.

    `env` defaults to `os.environ` but is overridable for testing.
    """

    def __init__(
        self,
        *,
        search_paths: list[Path],
        env_prefix: str,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not search_paths:
            raise ValueError("ProfileLoader requires at least one search path")
        self._search_paths = [Path(p) for p in search_paths]
        self._env_prefix = env_prefix
        self._env: Mapping[str, str] = env if env is not None else os.environ

    def load(self, name: str) -> Profile:
        merged = self._load_merged(name, _seen=set())
        merged["name"] = name
        return Profile.model_validate(merged)

    def names(self) -> list[str]:
        """All profile names discoverable across `search_paths`.

        Names are derived from `<name>.toml` filenames (stem). Duplicates
        across search paths are reported once (the earlier path wins, as
        with `load`).
        """
        seen: dict[str, None] = {}
        for path in self._search_paths:
            if not path.is_dir():
                continue
            for entry in sorted(path.glob("*.toml")):
                seen.setdefault(entry.stem, None)
        return list(seen)

    def load_all(self) -> dict[str, Profile]:
        return {name: self.load(name) for name in self.names()}

    def _load_merged(self, name: str, *, _seen: set[str]) -> dict[str, Any]:
        if name in _seen:
            chain = " -> ".join([*_seen, name])
            raise ProfileCycleError(f"Profile `extends` cycle detected: {chain}")
        _seen = {*_seen, name}

        raw = self._read_toml(name)
        parent_name = raw.pop("extends", None)
        if parent_name is not None and not isinstance(parent_name, str):
            raise ProfileLoaderError(
                f"Profile {name!r}: `extends` must be a string, got {type(parent_name).__name__}"
            )

        if parent_name:
            parent = self._load_merged(parent_name, _seen=_seen)
            merged = _deep_merge(parent, raw)
        else:
            merged = raw

        if self._env_prefix:
            _apply_env_overlay(
                merged,
                env=self._env,
                prefix=f"{self._env_prefix}_PROFILE_{_normalise_for_env(name)}_",
            )

        merged.pop("name", None)
        return merged

    def _read_toml(self, name: str) -> dict[str, Any]:
        for path in self._search_paths:
            candidate = path / f"{name}.toml"
            if candidate.is_file():
                with candidate.open("rb") as fh:
                    data = tomllib.load(fh)
                return dict(data)
        searched = ", ".join(str(p) for p in self._search_paths)
        raise ProfileNotFoundError(f"Profile {name!r} not found in any search path: {searched}")


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `over` onto `base`. Scalars and lists in `over` replace.
    Dicts merge key-by-key, recursively.
    """
    result: dict[str, Any] = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalise_for_env(name: str) -> str:
    """Profile / field name -> env-var segment.

    `claude-account-b` -> `CLAUDE_ACCOUNT_B`. Hyphens become underscores,
    everything uppercases.
    """
    return name.replace("-", "_").upper()


def _apply_env_overlay(
    target: MutableMapping[str, Any],
    *,
    env: Mapping[str, str],
    prefix: str,
) -> None:
    """Walk `env` and overlay any keys starting with `prefix` onto `target`.

    The path after `prefix` is split on `__` for nesting. Each segment is
    matched case-insensitively against the (possibly nested) dict keys
    already present in `target`. Segments that don't match an existing key
    create new entries — at the dict level, by introducing a string key
    normalised to lowercase.

    Values are coerced minimally: a JSON-parseable string becomes the parsed
    value (bool / number / null / list / dict); anything else stays a string.
    Pydantic will do the final coercion when the merged dict is validated.
    """
    import json

    for env_key, env_val in env.items():
        if not env_key.startswith(prefix):
            continue
        suffix = env_key[len(prefix) :]
        if not suffix:
            continue
        path = suffix.split(_NEST_DELIM)
        coerced = _coerce_env_value(env_val, json_loads=json.loads)
        _set_nested(target, path, coerced)


def _coerce_env_value(value: str, *, json_loads: Any) -> Any:
    """Best-effort coerce an env-string into a richer type.

    Treats `null`/`true`/`false`/numbers/JSON arrays/JSON objects literally.
    Anything else stays a string. Pydantic will reject mistypes downstream.
    """
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "{[" or stripped in {"true", "false", "null"} or _looks_numeric(stripped):
        try:
            return json_loads(stripped)
        except ValueError:
            return value
    return value


def _looks_numeric(s: str) -> bool:
    if not s:
        return False
    s = s.lstrip("-+")
    if not s:
        return False
    return s.replace(".", "", 1).isdigit() or _looks_scientific(s)


def _looks_scientific(s: str) -> bool:
    if "e" not in s and "E" not in s:
        return False
    mantissa, _, exponent = s.lower().partition("e")
    exponent = exponent.lstrip("-+")
    return mantissa.replace(".", "", 1).isdigit() and exponent.isdigit()


def _set_nested(target: MutableMapping[str, Any], path: list[str], value: Any) -> None:
    """Set `value` at `path` inside `target`, matching existing keys
    case-insensitively. Intermediate missing keys are created as dicts,
    with the normalised lowercase form of the env segment.
    """
    cursor: MutableMapping[str, Any] = target
    for segment in path[:-1]:
        match = _find_key_ci(cursor, segment)
        if match is None:
            match = segment.lower()
            cursor[match] = {}
        elif not isinstance(cursor[match], MutableMapping):
            cursor[match] = {}
        cursor = cursor[match]
    last = path[-1]
    match = _find_key_ci(cursor, last)
    cursor[match if match is not None else last.lower()] = value


def _find_key_ci(mapping: Mapping[str, Any], needle: str) -> str | None:
    """Find an existing key in `mapping` matching `needle` case-insensitively
    (and treating hyphens as equivalent to underscores, mirroring the env
    normalisation step).
    """
    needle_norm = needle.lower().replace("-", "_")
    for key in mapping:
        if key.lower().replace("-", "_") == needle_norm:
            return key
    return None
