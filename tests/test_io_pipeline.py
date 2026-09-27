"""normalize_dataset: robustness, ordering, schema, summary, out-of-area guard, no result drift."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from dataset_helpers import (
    ADDR_JUNK, ADDR_OK, ADDR_OTHER, ADDR_RAISES, assert_same_frames, exploding_stub, stub,
    synth_frame, write_csv,
)

from cali_address.io import (
    ColumnMapping, MappingError, MemorySink, Tunables, normalize_dataset, open_sink,
)
from cali_address.service import OUTPUT_COLUMNS, normalize_strict

CHUNK_SIZES = [1, 2, 6, 7, 8, 20_000]


def _run(source, mapping=None, *, normalizer=None, chunk_size=20_000, **kw):
    sink = MemorySink()
    summary = normalize_dataset(
        source, mapping or ColumnMapping(address_col="direccion"), sink,
        normalizer=normalizer or stub(), chunk_size=chunk_size, **kw,
    )
    return sink.frame, summary


# ---------------------------------------------------------------------------
# HARD RULE: identical rows -> identical results as the direct service call
# ---------------------------------------------------------------------------
def test_regression_pipeline_equals_direct_normalize_strict(tmp_path):
    frame = synth_frame(23)
    p = write_csv(tmp_path / "d.csv", frame)
    tun = Tunables()
    out, _ = _run(p, ColumnMapping(address_col="direccion", passthrough=False), tunables=tun, chunk_size=5)
    direct = normalize_strict(stub(), frame["direccion"].tolist(), **tun.as_kwargs())
    assert list(out.columns) == OUTPUT_COLUMNS
    assert_same_frames(out, direct)


def test_regression_non_default_tunables_are_forwarded(tmp_path):
    frame = synth_frame(9)
    p = write_csv(tmp_path / "d.csv", frame)
    tun = Tunables(min_struct=0.9, plate_tolerance=1, ambiguity_delta=0.0, gate_escalate=False)
    out, _ = _run(p, ColumnMapping(address_col="direccion", passthrough=False), tunables=tun, chunk_size=4)
    direct = normalize_strict(stub(), frame["direccion"].tolist(), **tun.as_kwargs())
    assert_same_frames(out, direct)


def test_tunables_defaults_match_the_legacy_cli():
    kw = Tunables().as_kwargs()
    assert kw["gate_escalate"] is True and kw["ambiguity_delta"] == 0.02
    assert kw["barrio_buffer_m"] == 1000.0 and kw["min_struct"] == 0.6 and kw["plate_tolerance"] == 0


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def test_passthrough_keeps_all_columns_then_output_schema(tmp_path):
    frame = synth_frame(4)
    out, _ = _run(write_csv(tmp_path / "d.csv", frame))
    expected = ["id", "direccion", "barrio", "extra_unknown"] + [c for c in OUTPUT_COLUMNS if c != "direccion_entrada"]
    assert list(out.columns) == expected
    assert out["extra_unknown"].tolist() == frame["extra_unknown"].tolist()
    assert out["id"].tolist() == frame["id"].tolist()


def test_passthrough_false_keeps_only_output_plus_id(tmp_path):
    frame = synth_frame(3)
    out, _ = _run(write_csv(tmp_path / "d.csv", frame),
                  ColumnMapping(address_col="direccion", id_col="id", passthrough=False))
    assert list(out.columns) == ["id"] + OUTPUT_COLUMNS


def test_passthrough_list_keeps_selected_columns(tmp_path):
    frame = synth_frame(3)
    out, _ = _run(write_csv(tmp_path / "d.csv", frame),
                  ColumnMapping(address_col="direccion", passthrough=["barrio"]))
    assert list(out.columns)[0] == "barrio" and "extra_unknown" not in out.columns


def test_source_columns_colliding_with_output_names_are_renamed(tmp_path):
    frame = synth_frame(3)
    frame["lat"] = "9.9"
    frame["estado"] = "viejo"
    out, _ = _run(write_csv(tmp_path / "d.csv", frame))
    assert out["lat_original"].tolist() == ["9.9"] * 3
    assert out["estado_original"].tolist() == ["viejo"] * 3
    assert set(out["estado"]) <= {"OK", "SIN_MATCH", "NO_PARSEABLE"}


def test_parts_mapping_builds_direccion_entrada_and_keeps_parts(tmp_path):
    frame = pd.DataFrame({"via": ["CL 5", "Carrera 100"], "num": ["# 38 - 20", "# 15-30"], "apto": [None, "Apto 301"]})
    out, _ = _run(write_csv(tmp_path / "p.csv", frame),
                  ColumnMapping(address_parts=["via", "num", "apto"]))
    assert list(out.columns)[:3] == ["via", "num", "apto"]
    assert out["direccion_entrada"].tolist() == ["CL 5 # 38 - 20", "Carrera 100 # 15-30 Apto 301"]
    assert out.loc[0, "estado"] == "OK"


def test_dataframe_and_iterable_sources_are_accepted():
    frame = synth_frame(5)
    out, summary = _run(iter([frame.iloc[:2], frame.iloc[2:]]))
    assert len(out) == 5 and summary["rows"] == 5
    out2, _ = _run(frame)
    assert_same_frames(out, out2)


# ---------------------------------------------------------------------------
# chunk boundaries, determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_size_never_changes_output(tmp_path, chunk_size):
    frame = synth_frame(7)
    p = write_csv(tmp_path / "d.csv", frame)
    reference, ref_summary = _run(p, chunk_size=20_000)
    out, summary = _run(p, chunk_size=chunk_size)
    assert len(out) == 7 and out["id"].tolist() == frame["id"].tolist()
    assert_same_frames(out, reference)
    assert summary["rows"] == 7 and summary["by_estado"] == ref_summary["by_estado"]


def test_output_is_deterministic(tmp_path):
    p = write_csv(tmp_path / "d.csv", synth_frame(11))
    a, _ = _run(p, chunk_size=3)
    b, _ = _run(p, chunk_size=3)
    assert_same_frames(a, b)


# ---------------------------------------------------------------------------
# hostile values
# ---------------------------------------------------------------------------
def test_hostile_address_values_never_abort_the_batch():
    frame = pd.DataFrame({"direccion": [
        ADDR_OK, None, np.nan, "", "   ", "\t\n", 12345, 3.5, dt.date(2024, 1, 2),
        pd.Timestamp("2024-01-02"), True, "x" * 10_000, "CL 5 # 38 - 20 " + "y" * 10_000,
        ADDR_JUNK, "'; DROP TABLE x; --", "<b>CL 5</b>", "Ñandú ​\u0000",
    ]})
    out, summary = _run(frame, chunk_size=4)
    assert len(out) == len(frame) and summary["rows"] == len(frame)
    assert set(out["estado"]) <= {"OK", "SIN_MATCH", "NO_PARSEABLE"}
    assert out.loc[0, "estado"] == "OK"
    assert out.loc[1, "estado"] == out.loc[2, "estado"] == out.loc[3, "estado"] == "SIN_MATCH"


def test_duplicate_and_missing_ids_are_preserved(tmp_path):
    frame = pd.DataFrame({"id": ["1", "1", "", None, "2"], "direccion": [ADDR_OK] * 5})
    out, _ = _run(frame, ColumnMapping(address_col="direccion", id_col="id"))
    assert out["id"].tolist()[:3] == ["1", "1", ""] and pd.isna(out["id"].iloc[3])
    assert len(out) == 5


# ---------------------------------------------------------------------------
# error isolation
# ---------------------------------------------------------------------------
def _poisoned_frame():
    return pd.DataFrame({
        "id": list("abcdefg"),
        "direccion": [ADDR_OK, ADDR_OK, ADDR_RAISES, ADDR_OK, ADDR_OTHER, ADDR_RAISES, ADDR_OK],
    })


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_bad_rows_become_error_rows_and_do_not_abort(chunk_size):
    out, summary = _run(_poisoned_frame(), normalizer=exploding_stub(), chunk_size=chunk_size)
    assert out["id"].tolist() == list("abcdefg")
    bad = out[out["estado"] == "ERROR"]
    assert bad["id"].tolist() == ["c", "f"]
    assert all(m.startswith("RuntimeError: ") and "exploded" in m for m in bad["motivo"])
    cadastral = [c for c in OUTPUT_COLUMNS
                 if c not in ("direccion_entrada", "estado", "motivo")]
    assert bad[cadastral].isna().all().all()
    assert summary["error"] == 2 and summary["by_estado"]["ERROR"] == 2
    assert summary["rows"] == 7 and summary["ok"] == 4
    assert summary["error_samples"] and "RuntimeError" in summary["error_samples"][0]


def test_good_rows_are_identical_with_or_without_poisoned_neighbours():
    frame = _poisoned_frame()
    good = frame[frame["direccion"] != ADDR_RAISES].reset_index(drop=True)
    with_bad, _ = _run(frame, normalizer=exploding_stub(), chunk_size=7)
    clean, _ = _run(good, normalizer=exploding_stub(), chunk_size=7)
    kept = with_bad[with_bad["estado"] != "ERROR"].reset_index(drop=True)
    assert_same_frames(kept, clean)


def test_error_message_is_short():
    class Verbose(type(exploding_stub())):
        def score_components(self, queries, k=20):
            if any(ADDR_RAISES in q for q in queries):
                raise ValueError("z" * 5000)
            return super().score_components(queries, k)

    out, _ = _run(_poisoned_frame(), normalizer=Verbose([{"direccion": ADDR_OK}]))
    assert max(len(m) for m in out.loc[out["estado"] == "ERROR", "motivo"]) < 300


def test_on_error_raise_propagates():
    with pytest.raises(RuntimeError):
        _run(_poisoned_frame(), normalizer=exploding_stub(), on_error="raise")


def test_invalid_on_error_value():
    with pytest.raises(ValueError):
        _run(synth_frame(2), on_error="explode")


def test_all_rows_bad_still_completes():
    frame = pd.DataFrame({"direccion": [ADDR_RAISES] * 4})
    out, summary = _run(frame, normalizer=exploding_stub(), chunk_size=2)
    assert (out["estado"] == "ERROR").all() and summary["error"] == 4 and summary["ok"] == 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_contents(tmp_path):
    frame = synth_frame(10)
    _, s = _run(write_csv(tmp_path / "d.csv", frame), chunk_size=4)
    assert s["rows"] == 10 and s["chunks"] == 3
    assert s["ok"] + s["sin_match"] + s["no_parseable"] + s["error"] + s["fuera_de_area"] == 10
    assert sum(s["by_estado"].values()) == 10
    assert s["by_nivel_precision"].get("predio", 0) + s["by_nivel_precision"].get("manzana", 0) \
        + s["by_nivel_precision"].get("direccion", 0) + s["by_nivel_precision"].get("via", 0) \
        + s["by_nivel_precision"].get("esquina", 0) == s["ok"]
    assert s["seconds"] >= 0 and s["rows_per_sec"] >= 0
    assert s["address_col"] == "direccion"


def test_progress_callback_gets_cumulative_rows():
    calls = []
    _run(synth_frame(7), chunk_size=3, progress=lambda done, chunk_no: calls.append((done, chunk_no)))
    assert calls == [(3, 1), (6, 2), (7, 3)]


def test_limit_processes_only_first_rows(tmp_path):
    p = write_csv(tmp_path / "d.csv", synth_frame(9))
    out, s = _run(p, chunk_size=4, limit=5)
    assert len(out) == 5 and s["rows"] == 5


# ---------------------------------------------------------------------------
# empty / tiny / mapping failures
# ---------------------------------------------------------------------------
def test_header_only_input_writes_header_and_zero_rows(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("id,direccion\n", encoding="utf-8")
    out_path = tmp_path / "o.csv"
    with open_sink(str(out_path)) as sink:
        s = normalize_dataset(str(p), ColumnMapping(), sink, normalizer=stub())
    back = pd.read_csv(out_path, encoding="utf-8-sig")
    assert len(back) == 0 and list(back.columns) == ["id", "direccion"] + [c for c in OUTPUT_COLUMNS if c != "direccion_entrada"]
    assert s["rows"] == 0 and s["rows_per_sec"] == 0


def test_single_row(tmp_path):
    out, s = _run(write_csv(tmp_path / "d.csv", synth_frame(1)))
    assert len(out) == 1 and s["ok"] == 1


def test_mapping_error_is_raised_before_the_normalizer_is_touched(tmp_path):
    n = exploding_stub()
    p = write_csv(tmp_path / "d.csv", pd.DataFrame({"a": ["x"], "b": ["y"]}))
    with pytest.raises(MappingError):
        _run(p, ColumnMapping(), normalizer=n)
    assert n.calls == 0


def test_lat_lon_are_informational_only(tmp_path):
    frame = synth_frame(6)
    frame["latitud"] = "3.4"
    frame["longitud"] = "-76.5"
    p = write_csv(tmp_path / "d.csv", frame)
    with_coords, _ = _run(p, ColumnMapping(address_col="direccion", lat_col="latitud", lon_col="longitud"))
    without, _ = _run(p, ColumnMapping(address_col="direccion"))
    assert_same_frames(with_coords, without)


def test_schema_drift_between_chunks_is_a_read_error():
    a = pd.DataFrame({"direccion": [ADDR_OK]})
    b = pd.DataFrame({"direccion": [ADDR_OK], "surprise": ["x"]})
    from cali_address.io import SourceReadError

    with pytest.raises(SourceReadError, match="columns"):
        _run(iter([a, b]))


# ---------------------------------------------------------------------------
# out-of-area guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [
    "Cali", "CALI", "cali", " Cali ", "Santiago de Cali", "SANTIAGO DE CALI", "santiago  de   cali",
    "CALI - VALLE", "Cali, Valle", "Cali (Valle del Cauca)", "Santiago de Cali - Valle del Cauca",
    "76001", "Cáli",
])
def test_municipality_cali_variants_are_matched_normally(value):
    frame = pd.DataFrame({"direccion": [ADDR_OK], "mun": [value]})
    out, s = _run(frame, ColumnMapping(address_col="direccion", municipality_col="mun"))
    assert out.loc[0, "estado"] == "OK", value


@pytest.mark.parametrize("value", ["Bogotá", "Palmira", "Jamundí", "Yumbo", "Cartago", "Caliente", "Medellín", "11001"])
def test_other_municipalities_are_marked_out_of_area_without_matching(value):
    frame = pd.DataFrame({"direccion": [ADDR_OK], "mun": [value]})
    n = exploding_stub()
    out, s = _run(frame, ColumnMapping(address_col="direccion", municipality_col="mun"), normalizer=n)
    assert out.loc[0, "estado"] == "FUERA_DE_AREA" and "Cali" in out.loc[0, "motivo"]
    assert out.loc[0, "numero_predial_nacional"] is None or pd.isna(out.loc[0, "numero_predial_nacional"])
    assert n.calls == 0 and s["fuera_de_area"] == 1


@pytest.mark.parametrize("value", [None, np.nan, "", "   ", "N/A", "sin dato", "-"])
def test_empty_or_placeholder_municipality_is_not_guarded(value):
    frame = pd.DataFrame({"direccion": [ADDR_OK], "mun": [value]})
    out, _ = _run(frame, ColumnMapping(address_col="direccion", municipality_col="mun"))
    assert out.loc[0, "estado"] == "OK"


def test_guard_is_off_without_municipality_col(tmp_path):
    frame = pd.DataFrame({"direccion": [ADDR_OK], "mun": ["Bogotá"]})
    out, _ = _run(frame)
    assert out.loc[0, "estado"] == "OK"


def test_guard_preserves_order_and_neighbours():
    frame = pd.DataFrame({
        "id": list("abcd"), "direccion": [ADDR_OK] * 4, "mun": ["Cali", "Bogotá", "Cali", "Palmira"],
    })
    out, s = _run(frame, ColumnMapping(address_col="direccion", municipality_col="mun"), chunk_size=3)
    assert out["id"].tolist() == list("abcd")
    assert out["estado"].tolist() == ["OK", "FUERA_DE_AREA", "OK", "FUERA_DE_AREA"]
    direct = normalize_strict(stub(), [ADDR_OK], **Tunables().as_kwargs())
    assert out.loc[0, "direccion_normalizada"] == direct.loc[0, "direccion_normalizada"]
