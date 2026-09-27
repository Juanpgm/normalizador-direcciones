"""``--config FILE.toml``: the same options as the command line.

Keys are the long option names with ``-`` or ``_`` (``address-col`` == ``address_col``).
Either a flat table, or the same keys grouped in any sub-tables (``[input]``,
``[columns]``, ``[tunables]``...), which are flattened. Precedence, lowest to
highest: built-in defaults, config file, command line.

Unknown keys are rejected (a typo would otherwise silently do nothing).
"""

from __future__ import annotations

import os
import tomllib

from .errors import ConfigError

_STR = "str"
_INT = "int"
_POS_INT = "pos_int"
_NON_NEG_INT = "non_neg_int"
_FLOAT = "float"
_BOOL = "bool"
_LIST = "list"

#: option name -> value kind. The CLI defines the flags; this is the config-side contract.
OPTION_KINDS: dict[str, str] = {
    "input": _STR, "output": _STR, "format": _STR, "output_format": _STR,
    "sheet": _STR, "header_row": _NON_NEG_INT, "delimiter": _STR, "encoding": _STR,
    "address_col": _STR, "address_parts": _LIST, "parts_sep": _STR, "id_col": _STR,
    "municipality_col": _STR, "lat_col": _STR, "lon_col": _STR, "keep_columns": _LIST,
    "chunk_size": _POS_INT, "table": _STR,
    "artifacts_dir": _STR, "basemaps": _STR, "device": _STR, "threshold": _FLOAT,
    "min_struct": _FLOAT, "plate_tolerance": _INT, "ambiguity_delta": _FLOAT,
    "barrio_buffer": _FLOAT, "zone_buffer": _FLOAT, "gate_escalate": _BOOL,
    "gazetteer": _BOOL, "no_gazetteer": _BOOL, "soft_rules": _LIST, "max_soft": _NON_NEG_INT,
    "gate_fallback": _BOOL, "on_error": _STR, "dry_run": _POS_INT, "summary_json": _STR,
}

_RANGES = {
    "min_struct": (0.0, 1.0), "threshold": (0.0, 1.0), "plate_tolerance": (0, 2),
    "ambiguity_delta": (0.0, 0.2), "barrio_buffer": (0.0, 1e9), "zone_buffer": (0.0, 1e9),
}


def _coerce(key: str, value):
    kind = OPTION_KINDS[key]
    bad = ConfigError(f"config option {key!r}: invalid value {value!r} (expected {kind.replace('_', ' ')})")
    if kind == _BOOL:
        if not isinstance(value, bool):
            raise bad
        return value
    if kind == _STR:
        if not isinstance(value, str):
            raise bad
        return value
    if kind == _LIST:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
        raise bad
    if kind in (_INT, _POS_INT, _NON_NEG_INT):
        if isinstance(value, bool) or not isinstance(value, int):
            raise bad
        if kind == _POS_INT and value < 1:
            raise ConfigError(f"config option {key!r} must be >= 1 (got {value})")
        if kind == _NON_NEG_INT and value < 0:
            raise ConfigError(f"config option {key!r} must be >= 0 (got {value})")
    if kind == _FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad
        value = float(value)
    if key in _RANGES:
        lo, hi = _RANGES[key]
        if not lo <= value <= hi:
            raise ConfigError(f"config option {key!r} must be between {lo} and {hi} (got {value})")
    return value


def _flatten(table: dict, out: dict) -> None:
    for key, value in table.items():
        if isinstance(value, dict):
            _flatten(value, out)
        else:
            out[str(key).replace("-", "_")] = value


def load_config(path: str) -> dict:
    """Parse and validate a TOML config file; returns ``{option: value}``."""
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    flat: dict = {}
    _flatten(raw, flat)
    unknown = sorted(set(flat) - set(OPTION_KINDS))
    if unknown:
        raise ConfigError(
            f"unknown config option(s) in {path}: {', '.join(unknown)}. "
            f"Valid options: {', '.join(sorted(OPTION_KINDS))}"
        )
    return {key: _coerce(key, value) for key, value in flat.items()}


#: Options that hold a filesystem path. In a config file a relative one is relative to that file's directory.
PATH_OPTIONS = ("input", "output", "summary_json", "artifacts_dir", "basemaps")


def resolve_config_paths(options: dict, base_dir: str) -> dict:
    """Copy of ``options`` with relative path options made absolute against ``base_dir``.

    URLs (anything with ``://``: http(s), SQLAlchemy) and empty values are left as written; a leading ``~``
    is expanded to the home directory; absolute paths are unchanged. Command line values never go through here (they stay relative to the CWD).
    """
    resolved = dict(options)
    for key in PATH_OPTIONS:
        value = resolved.get(key)
        if isinstance(value, str) and value.startswith("~"):
            value = resolved[key] = os.path.normpath(os.path.expanduser(value))  # "~" / "~/x": the user's home, not <cfgdir>/~
        if not isinstance(value, str) or not value or "://" in value or os.path.isabs(value):
            continue
        resolved[key] = os.path.normpath(os.path.join(base_dir, value))
    return resolved


def merge_options(defaults: dict, file_options: dict, cli_options: dict) -> dict:
    """defaults < config file < command line. A CLI value of ``None`` means "not given"."""
    merged = dict(defaults)
    merged.update(file_options)
    merged.update({k: v for k, v in cli_options.items() if v is not None})
    return merged
