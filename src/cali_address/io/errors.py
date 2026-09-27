"""Exception hierarchy for the dataset IO layer.

Two families, mapped to CLI exit codes by :mod:`cali_address.cli`:

* :class:`UsageError` (exit 2): the caller asked for something that cannot work
  (unknown format, bad column mapping, malformed config, missing optional
  dependency). Fixing the command line fixes it.
* :class:`DatasetIOError` (exit 3): the request was fine but reading or writing
  failed (missing file, corrupt file, unreachable database, unwritable output).
"""

from __future__ import annotations


class DatasetError(Exception):
    """Base class for every expected, user-facing failure of the IO layer."""


class UsageError(DatasetError):
    """The request itself is invalid."""


class UnsupportedFormatError(UsageError):
    """Unknown or unsupported input/output format."""


class MissingDependencyError(UsageError):
    """An optional dependency needed for this format is not installed."""


class MappingError(UsageError, ValueError):
    """The column mapping does not fit the table."""


class ConfigError(UsageError):
    """The ``--config`` file is missing, malformed or holds invalid values."""


class DatasetIOError(DatasetError):
    """Reading or writing failed."""


class SourceReadError(DatasetIOError):
    """The input could not be read."""


class SinkWriteError(DatasetIOError):
    """The output could not be written."""
