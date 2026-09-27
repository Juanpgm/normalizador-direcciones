"""Column mapping: which input columns mean what, validated against the real schema.

:class:`ColumnMapping` is the user's intent (all fields optional);
:meth:`ColumnMapping.resolve` checks it against the columns of the first chunk and
returns a :class:`ResolvedMapping` with concrete column names, or raises
:class:`MappingError` with a message that lists the columns that were detected.

Address column resolution reuses ``cali_address.tables.resolve_address_column``
(synonym scoring). Its "several columns could be the address" outcome is NOT
turned into a guess: the original ``AddressColumnError`` propagates so the caller
can ask the user to choose.

Coordinates (``lat_col`` / ``lon_col``) are validated and reported but never
influence matching: they are never a correctness criterion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from .errors import MappingError


def _missing(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _part_text(value) -> str:
    if _missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _find(name: str, columns: list[str]) -> str | None:
    """Exact match first, then case-insensitive on stripped names."""
    if name in columns:
        return name
    lowered = name.strip().lower()
    for col in columns:
        if col.strip().lower() == lowered:
            return col
    return None


def _split_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(v) for v in value]


@dataclass
class ColumnMapping:
    """Which columns hold the address, the id, the municipality and (informational) the coordinates."""

    address_col: str | None = None
    address_parts: Sequence[str] | None = None
    parts_sep: str = " "
    id_col: str | None = None
    municipality_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    #: ``True`` keeps every input column, ``False`` none, a list keeps just those.
    passthrough: bool | Sequence[str] = True

    @classmethod
    def from_options(cls, *, address_col=None, address_parts=None, parts_sep=" ", id_col=None,
                     municipality_col=None, lat_col=None, lon_col=None, keep_columns=None) -> "ColumnMapping":
        """Build from CLI/config values: ``address_parts`` may be ``"a,b,c"``; ``keep_columns`` is
        ``None``/``"all"`` (everything), ``"none"`` or a comma separated list."""
        passthrough: bool | list[str] = True
        if keep_columns is not None:
            if isinstance(keep_columns, str) and keep_columns.strip().lower() == "all":
                passthrough = True
            elif isinstance(keep_columns, str) and keep_columns.strip().lower() == "none":
                passthrough = False
            else:
                passthrough = _split_list(keep_columns) or []
        return cls(
            address_col=address_col or None, address_parts=_split_list(address_parts),
            parts_sep=" " if parts_sep is None else parts_sep, id_col=id_col or None,
            municipality_col=municipality_col or None, lat_col=lat_col or None, lon_col=lon_col or None,
            passthrough=passthrough,
        )

    # ------------------------------------------------------------------
    def resolve(self, columns: Sequence) -> "ResolvedMapping":
        from ..tables import AddressColumnError, resolve_address_column  # lazy

        cols = [str(c) for c in columns]
        if not cols:
            raise MappingError("the input has no columns")
        shown = ", ".join(cols)

        def need(name: str, role: str) -> str:
            hit = _find(str(name), cols)
            if hit is None:
                raise MappingError(f"{role} column {name!r} not found. Detected columns: {shown}")
            return hit

        if self.address_col and self.address_parts is not None:
            raise MappingError("use either --address-col or --address-parts, not both")

        parts: list[str] | None = None
        address_col: str | None = None
        if self.address_parts is not None:
            if len(self.address_parts) == 0:
                raise MappingError("address_parts is empty: name at least one column")
            parts = [need(p, "address part") for p in self.address_parts]
        elif self.address_col:
            address_col = need(self.address_col, "address")
        else:
            try:
                address_col = resolve_address_column(pd.DataFrame(columns=cols), None)
            except AddressColumnError as exc:
                if exc.candidates:
                    raise  # ambiguous: the user must choose, never guess
                raise MappingError(
                    f"no column looks like an address. Detected columns: {shown}. "
                    "Use --address-col NAME, or --address-parts A,B,C to join several columns."
                ) from exc

        if bool(self.lat_col) != bool(self.lon_col):
            raise MappingError("give both lat_col and lon_col, or neither (coordinates are informational only)")

        id_col = need(self.id_col, "id") if self.id_col else None
        municipality_col = need(self.municipality_col, "municipality") if self.municipality_col else None
        lat_col = need(self.lat_col, "latitude") if self.lat_col else None
        lon_col = need(self.lon_col, "longitude") if self.lon_col else None

        if self.passthrough is True:
            passthrough = list(cols)
        elif self.passthrough is False:
            passthrough = []
        else:
            passthrough = [need(c, "passthrough") for c in self.passthrough]
        if id_col and id_col not in passthrough:
            passthrough.insert(0, id_col)

        return ResolvedMapping(
            address_col=address_col, address_parts=parts, parts_sep=self.parts_sep, id_col=id_col,
            municipality_col=municipality_col, lat_col=lat_col, lon_col=lon_col, passthrough_cols=passthrough,
            columns=cols,
        )


@dataclass(frozen=True)
class ResolvedMapping:
    address_col: str | None
    address_parts: list[str] | None
    parts_sep: str
    id_col: str | None
    municipality_col: str | None
    lat_col: str | None
    lon_col: str | None
    passthrough_cols: list[str]
    columns: list[str] = field(default_factory=list)

    def build_addresses(self, frame: pd.DataFrame) -> list:
        """The raw address of every row: the column as-is, or the joined parts (blanks skipped)."""
        if self.address_parts is None:
            return frame[self.address_col].tolist()
        texts = [[_part_text(v) for v in frame[p].tolist()] for p in self.address_parts]
        return [self.parts_sep.join(t for t in row if t) for row in zip(*texts)] if len(frame) else []

    def describe(self) -> dict:
        return {
            "address_col": self.address_col, "address_parts": self.address_parts,
            "id_col": self.id_col, "municipality_col": self.municipality_col,
            "lat_col": self.lat_col, "lon_col": self.lon_col,
        }
