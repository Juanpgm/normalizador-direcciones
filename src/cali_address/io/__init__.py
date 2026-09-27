"""Dataset IO for the Cali address normalizer: read any table, normalize, write anywhere.

Public surface (see ``docs/any-dataset.md``):

* :func:`read_table` / :class:`TableChunks` / :func:`infer_format` - chunked readers
* :func:`open_sink` / :class:`Sink` / :class:`MemorySink` - chunked, atomic writers
* :class:`ColumnMapping` - which columns hold the address, id, municipality
* :func:`normalize_dataset` / :class:`Tunables` - the robust pipeline
* :mod:`cali_address.io.config` - the ``--config`` TOML layer
"""

from .config import load_config, merge_options, resolve_config_paths
from .errors import (
    ConfigError,
    DatasetError,
    DatasetIOError,
    MappingError,
    MissingDependencyError,
    SinkWriteError,
    SourceReadError,
    UnsupportedFormatError,
    UsageError,
)
from .mapping import ColumnMapping, ResolvedMapping
from .readers import (
    DEFAULT_CHUNK_SIZE,
    TableChunks,
    describe_reader_formats,
    infer_format,
    list_reader_formats,
    read_table,
    register_reader,
)
from .writers import (
    MemorySink,
    Sink,
    describe_writer_formats,
    infer_sink_format,
    list_writer_formats,
    open_sink,
    register_writer,
)

__all__ = [
    "ColumnMapping", "ConfigError", "DEFAULT_CHUNK_SIZE", "DatasetError", "DatasetIOError", "MappingError",
    "MemorySink", "MissingDependencyError", "ResolvedMapping", "Sink", "SinkWriteError", "SourceReadError",
    "TableChunks", "Tunables", "UnsupportedFormatError", "UsageError", "describe_reader_formats",
    "describe_writer_formats", "infer_format", "infer_sink_format", "list_reader_formats",
    "list_writer_formats", "load_config", "merge_options", "normalize_dataset", "open_sink", "read_table",
    "register_reader", "register_writer", "resolve_config_paths",
]


def __getattr__(name: str):
    # The pipeline imports the model stack (torch via cali_address.service); keep `import cali_address.io` light.
    if name in ("normalize_dataset", "Tunables"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
