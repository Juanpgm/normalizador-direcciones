"""ColumnMapping: resolution, validation against the first chunk's schema, address building."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cali_address.io import ColumnMapping, MappingError
from cali_address.service import AddressColumnError

COLS = ["id", "direccion", "municipio", "barrio"]


def test_auto_detects_address_column():
    r = ColumnMapping().resolve(COLS)
    assert r.address_col == "direccion" and r.address_parts is None


def test_explicit_address_column_case_insensitive():
    assert ColumnMapping(address_col="DIRECCION").resolve(COLS).address_col == "direccion"


def test_missing_explicit_address_column_lists_detected_columns():
    with pytest.raises(MappingError) as exc:
        ColumnMapping(address_col="calle_x").resolve(COLS)
    msg = str(exc.value)
    assert "calle_x" in msg and all(c in msg for c in COLS)


def test_no_address_like_column_is_actionable():
    with pytest.raises(MappingError) as exc:
        ColumnMapping().resolve(["a", "b", "c"])
    msg = str(exc.value)
    assert "--address-col" in msg and "a, b, c" in msg.replace("'", "")


def test_ambiguous_columns_keep_the_user_disambiguation_error():
    with pytest.raises(AddressColumnError) as exc:
        ColumnMapping().resolve(["id", "direccion", "domicilio"])
    assert {c["column"] for c in exc.value.candidates} >= {"direccion", "domicilio"}


def test_address_col_and_parts_are_mutually_exclusive():
    with pytest.raises(MappingError, match="both"):
        ColumnMapping(address_col="direccion", address_parts=["a", "b"]).resolve(["direccion", "a", "b"])


def test_parts_must_exist_and_be_non_empty():
    with pytest.raises(MappingError, match="via2"):
        ColumnMapping(address_parts=["via", "via2"]).resolve(["via", "num"])
    with pytest.raises(MappingError):
        ColumnMapping(address_parts=[]).resolve(["via"])


def test_parts_mapping_resolves_without_address_col():
    r = ColumnMapping(address_parts=["VIA", "num"]).resolve(["via", "num", "x"])
    assert r.address_parts == ["via", "num"] and r.address_col is None


@pytest.mark.parametrize("field", ["id_col", "municipality_col", "lat_col", "lon_col"])
def test_optional_columns_must_exist(field):
    kwargs = {"address_col": "direccion", field: "nope"}
    if field in ("lat_col", "lon_col"):
        kwargs = {"address_col": "direccion", "lat_col": "nope", "lon_col": "nope"}
    with pytest.raises(MappingError, match="nope"):
        ColumnMapping(**kwargs).resolve(COLS)


def test_lat_lon_must_come_together():
    with pytest.raises(MappingError, match="lat"):
        ColumnMapping(address_col="direccion", lat_col="id").resolve(COLS)


def test_no_columns_at_all():
    with pytest.raises(MappingError, match="no columns"):
        ColumnMapping().resolve([])


def test_passthrough_list_must_exist():
    with pytest.raises(MappingError, match="zzz"):
        ColumnMapping(address_col="direccion", passthrough=["barrio", "zzz"]).resolve(COLS)


def test_from_options_accepts_comma_string_parts():
    m = ColumnMapping.from_options(address_parts="via, numero ,complemento")
    assert m.address_parts == ["via", "numero", "complemento"]


# ---------------------------------------------------------------------------
# building the address text
# ---------------------------------------------------------------------------
def test_build_addresses_single_column_is_untouched():
    frame = pd.DataFrame({"direccion": ["CL 5 # 38 - 20", None, "  x  "]})
    r = ColumnMapping(address_col="direccion").resolve(["direccion"])
    out = r.build_addresses(frame)
    assert out[0] == "CL 5 # 38 - 20" and out[2] == "  x  "
    assert out[1] is None or (isinstance(out[1], float) and np.isnan(out[1]))


def test_build_addresses_from_parts_skips_blanks_and_formats_numbers():
    frame = pd.DataFrame({
        "via": ["CL 5", "Carrera 100", None, "", "  "],
        "numero": [38.0, "# 15-30", np.nan, "", None],
        "compl": ["Apto 301", None, None, "", ""],
    })
    r = ColumnMapping(address_parts=["via", "numero", "compl"]).resolve(list(frame.columns))
    assert r.build_addresses(frame) == ["CL 5 38 Apto 301", "Carrera 100 # 15-30", "", "", ""]


def test_build_addresses_custom_separator():
    frame = pd.DataFrame({"a": ["CL 5", "CL 6"], "b": ["# 38 - 20", None]})
    r = ColumnMapping(address_parts=["a", "b"], parts_sep=", ").resolve(["a", "b"])
    assert r.build_addresses(frame) == ["CL 5, # 38 - 20", "CL 6"]


def test_build_addresses_empty_frame():
    r = ColumnMapping(address_col="direccion").resolve(["direccion"])
    assert list(r.build_addresses(pd.DataFrame({"direccion": []}))) == []
