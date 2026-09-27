"""Tests for the place gazetteer (barrio / vereda / comuna / corregimiento).

Ground truth comes from the two IDESC basemaps under ``basemaps/``; the counts
asserted here were measured from those files (see ``test_load_counts``) so a
silently swapped basemap fails loudly instead of degrading detection.

Documented precedence decisions exercised below
-----------------------------------------------
* ``Navarro`` is simultaneously a corregimiento (id 51), the cabecera vereda of
  that corregimiento and part of the barrio ``Navarro - La Chanca``. For an
  unlabeled mention the gazetteer prefers **corregimiento over vereda over
  barrio-substring**, because the corregimiento polygon is the superset and the
  geographic gate must never be tighter than the evidence justifies.
* A detected barrio fills the comuna from its parent; an explicit comuna that
  disagrees is kept and flagged with the ``comuna_barrio_conflict`` note.
* There is **no** ``Danubio`` barrio in the official layer, so the Danubio
  fixture asserts the address core is isolated (``cleaned_text`` parses where the
  raw text did not) with no barrio detected.
"""

from __future__ import annotations

import os
import pickle
import sys

import pytest
from shapely import Point

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from cali_address.gazetteer import Candidate, Gazetteer, gate_candidates  # noqa: E402
from cali_address.parser import parse_address  # noqa: E402

BASEMAPS = os.path.join(PROJECT_ROOT, "basemaps")

# IDESC reports 3.4278840 / -76.5448877 as San Fernando Nuevo, comuna 19.
SAN_FERNANDO_LON, SAN_FERNANDO_LAT = -76.5448877, 3.4278840


@pytest.fixture(scope="module")
def gaz() -> Gazetteer:
    return Gazetteer.load(BASEMAPS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_counts(gaz):
    """Unique entries per kind, after folding accents and unioning duplicates."""
    assert len(gaz.by_kind("barrio")) == 339
    assert len(gaz.by_kind("vereda")) == 91
    assert len(gaz.by_kind("comuna")) == 22
    assert len(gaz.by_kind("corregimiento")) == 15


def test_duplicate_names_are_unioned_not_dropped(gaz):
    """`Las Nieves` exists in El Saladito and in Felidia; both polygons survive."""
    entry = gaz.lookup("vereda", "Las Nieves")
    assert entry is not None
    assert entry.codes == ("6007", "5903")
    assert entry.parents == ("El Saladito", "Felidia")
    assert entry.parent is None  # ambiguous parent is reported as such
    rows = [r for r in gaz.rows if r.kind == "vereda" and r.name == "Las Nieves"]
    assert len(rows) == 2
    assert entry.geometry.area == pytest.approx(sum(r.geometry.area for r in rows), rel=1e-9)


def test_repeated_identical_barrio_rows_collapse_to_one_code(gaz):
    """`Vista Hermosa` is duplicated verbatim in the basemap (same comuna and id)."""
    entry = gaz.lookup("barrio", "Vista Hermosa")
    assert entry.codes == ("0102",)
    assert entry.parents == ("01",)


def test_accent_only_duplicates_collapse(gaz):
    """`La María` (Golondrinas) and `La Maria` (Pance) fold to a single key."""
    entry = gaz.lookup("vereda", "La Maria")
    assert entry is not None
    assert len(entry.codes) == 2


def test_cabecera_suffix_stripped_from_key(gaz):
    entry = gaz.lookup("vereda", "Pance (Cabecera)")
    assert entry is not None
    assert entry.key == "PANCE"


def test_barrio_entry_carries_official_name_and_codes(gaz):
    entry = gaz.lookup("barrio", "siloe")
    assert entry.name == "Siloé"
    assert entry.codes == ("2003",)
    assert entry.parent == "20"


def test_vereda_parent_is_its_corregimiento(gaz):
    entry = gaz.lookup("vereda", "El Pinar")
    assert entry.parent == "La Castilla"


def test_comuna_entry_indexed_by_number(gaz):
    entry = gaz.comuna_entry(19)
    assert entry is not None
    assert entry.name == "Comuna 19"
    assert entry.geometry.contains(Point(SAN_FERNANDO_LON, SAN_FERNANDO_LAT))


def test_urban_rows_ignore_stray_vereda_columns(gaz):
    """`Vista Hermosa` (urban) must not leak the spatial-join vereda it carries."""
    assert gaz.lookup("vereda", "Pilas del Cabuyal").parent == "Los Andes"
    assert gaz.lookup("barrio", "Terron Colorado").parent == "01"


def test_load_writes_and_reuses_pickle_cache(tmp_path):
    cache = tmp_path / "gaz.pkl"
    first = Gazetteer.load(BASEMAPS, cache_path=str(cache))
    assert cache.exists()
    with open(cache, "rb") as fh:
        blob = pickle.load(fh)
    assert "mtimes" in blob
    second = Gazetteer.load(BASEMAPS, cache_path=str(cache))
    assert second.from_cache is True
    assert first.from_cache is False
    assert len(second.entries) == len(first.entries)


def test_stale_cache_is_rebuilt(tmp_path):
    cache = tmp_path / "gaz.pkl"
    Gazetteer.load(BASEMAPS, cache_path=str(cache))
    with open(cache, "rb") as fh:
        blob = pickle.load(fh)
    blob["mtimes"] = {k: 0.0 for k in blob["mtimes"]}
    with open(cache, "wb") as fh:
        pickle.dump(blob, fh)
    rebuilt = Gazetteer.load(BASEMAPS, cache_path=str(cache))
    assert rebuilt.from_cache is False
    assert len(rebuilt.by_kind("barrio")) == 339


def test_corrupt_cache_is_rebuilt(tmp_path):
    cache = tmp_path / "gaz.pkl"
    cache.write_bytes(b"not a pickle")
    gaz = Gazetteer.load(BASEMAPS, cache_path=str(cache))
    assert gaz.from_cache is False
    assert len(gaz.by_kind("comuna")) == 22


def test_missing_basemaps_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Gazetteer.load(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Detection - explicit labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Barrio Siloé calle 1 # 2-3", "Siloé"),
        ("BARRIO SILOE", "Siloé"),
        ("brr. San Fernando", "San Fernando Nuevo"),
        ("B/ Meléndez calle 5 # 1-2", "Meléndez"),
        ("Urbanizacion La Flora cl 44 # 3-4", "La Flora"),
    ],
)
def test_barrio_labels(gaz, raw, expected):
    det = gaz.detect(raw)
    assert det.barrio == expected, det


def test_sector_label_consumes_the_word_sector(gaz):
    """`Sector Meléndez` is itself a barrio, so either reading is defensible."""
    assert gaz.detect("Sector Melendez").barrio in ("Meléndez", "Sector Meléndez")


def test_brr_san_fernando_is_comuna_19(gaz):
    det = gaz.detect("brr. San Fernando cl 5 # 39-20")
    assert det.barrio.startswith("San Fernando")
    assert det.comuna == 19


def test_vereda_label(gaz):
    assert gaz.detect("VDA El Pinar").vereda == "El Pinar"


def test_ambiguous_vereda_label_still_resolves_to_a_reforma_vereda(gaz):
    """`La Reforma` is not an official vereda name; two veredas contain it."""
    assert "Reforma" in gaz.detect("vereda la reforma").vereda


@pytest.mark.parametrize(
    "raw, correg, vereda",
    [
        ("Corregimiento los andes, vereda la reforma, casa 117", "Los Andes",
         "El Mango - La Reforma"),
        ("corregimiento la buitrera, vereda la reforma", "La Buitrera",
         "Acueducto de la Reforma"),
    ],
)
def test_vereda_disambiguated_by_explicit_corregimiento(gaz, raw, correg, vereda):
    det = gaz.detect(raw)
    assert (det.corregimiento, det.vereda) == (correg, vereda)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Corregimiento los andes", "Los Andes"),
        ("CGTO Pance", "Pance"),
        ("Correg. La Buitrera", "La Buitrera"),
    ],
)
def test_corregimiento_labels(gaz, raw, expected):
    assert gaz.detect(raw).corregimiento == expected


def test_vereda_implies_its_corregimiento(gaz):
    det = gaz.detect("vereda El Pinar")
    assert det.vereda == "El Pinar"
    assert det.corregimiento == "La Castilla"


@pytest.mark.parametrize("raw, expected", [("COMUNA 15", 15), ("comuna 3", 3), ("Comuna  22", 22)])
def test_comuna_label(gaz, raw, expected):
    assert gaz.detect(raw).comuna == expected


@pytest.mark.parametrize("raw", ["COMUNA 0", "comuna 45", "comuna 99"])
def test_out_of_range_comuna_is_rejected(gaz, raw):
    det = gaz.detect(raw)
    assert det.comuna is None
    assert "comuna_out_of_range" in det.notes


def test_comuna_number_is_not_swallowed_from_a_longer_run(gaz):
    """`COMUNA 190` is not comuna 19."""
    det = gaz.detect("cl 5 # 3-4 COMUNA 190")
    assert det.comuna is None


def test_label_without_a_matching_name_detects_nothing(gaz):
    det = gaz.detect("Barrio Zzzqqq calle 1 # 2-3")
    assert det.barrio is None


def test_label_does_not_match_a_far_away_name(gaz):
    """rapidfuzz cutoff: `Barrio Silvania` must not become `Siloé`."""
    assert gaz.detect("Barrio Silvania").barrio != "Siloé"


# ---------------------------------------------------------------------------
# Detection - accents, case, encoding damage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["SILOE", "siloé", "Siloe", "  siloe  "])
def test_accent_and_case_insensitive_segment_match(gaz, raw):
    assert gaz.detect(raw).barrio == "Siloé"


def test_unaccented_query_matches_accented_official_name(gaz):
    assert gaz.detect("cl 5 # 1-2, Melendez").barrio == "Meléndez"


def test_mojibake_does_not_crash_and_still_uses_intact_tokens(gaz):
    det = gaz.detect("Urbanizaci�n La Flora\nCalle 44 # 3-4")
    assert det.barrio == "La Flora"
    assert parse_address(det.cleaned_text).parse_ok


def test_mojibake_inside_the_name_does_not_crash(gaz):
    det = gaz.detect("Barrio Silo�")
    assert det.notes is not None  # no exception; detection may or may not fire


# ---------------------------------------------------------------------------
# Detection - segments
# ---------------------------------------------------------------------------


def test_segment_navarro_prefers_corregimiento(gaz):
    """Documented precedence: corregimiento > vereda > barrio for `Navarro`."""
    det = gaz.detect("Cl. 55b # 47-55, Navarro, Cali, Valle del Cauca")
    assert det.corregimiento == "Navarro"
    assert det.barrio is None
    assert det.cleaned_text.strip().rstrip(",") == "Cl. 55b # 47-55"


@pytest.mark.parametrize(
    "raw",
    [
        "Cl. 16 #35-57, Santa Elena, Cali, Valle del Cauca",
        "Cl. 16 #35-57 / Santa Elena / Cali",
        "Cl. 16 #35-57 | Santa Elena | Cali",
        "Cl. 16 #35-57; Santa Elena",
        "Cl. 16 #35-57\nSanta Elena\nCali",
    ],
)
def test_segment_separators(gaz, raw):
    det = gaz.detect(raw)
    assert det.barrio == "Santa Elena", det
    assert "SANTA ELENA" not in det.cleaned_text.upper()


def test_city_and_department_noise_never_matches_a_place(gaz):
    det = gaz.detect("Cl. 16 #35-57, Cali, Valle del Cauca, Colombia")
    assert (det.barrio, det.vereda, det.comuna, det.corregimiento) == (None, None, None, None)


def test_santiago_de_cali_noise_is_dropped(gaz):
    det = gaz.detect("Cl. 16 #35-57, Santa Elena, Santiago de Cali")
    assert det.barrio == "Santa Elena"


def test_short_segment_is_not_fuzzy_matched(gaz):
    """Segments with fewer than 5 letters only match exactly."""
    det = gaz.detect("Cl. 16 #35-57, AB")
    assert det.barrio is None


def test_fuzzy_match_prefers_the_name_that_covers_the_whole_mention(gaz):
    """`token_set_ratio` scores a subset 100, so coverage must break the tie.

    Without it `Nueva Granada` resolves to the barrio `Granada` in comuna 2
    instead of `Urbanización Nueva Granada` in comuna 19, and the geographic gate
    then rejects every correct candidate.
    """
    det = gaz.detect("Cra. 38 # 4C-30, Nueva Granada, Cali, Valle del Cauca")
    assert det.barrio == "Urbanización Nueva Granada"
    assert det.comuna == 19


@pytest.mark.parametrize(
    "raw",
    [
        "Avenida 2B Norte #34AN-55 Bloque No 1 Conjunto Residencial Zoila",
        "Calle 2 62 b 19 Conjunto residencial manantial torre A",
        "Cl 5 # 3-4 conjunto residencial",
    ],
)
def test_a_generic_word_does_not_match_a_longer_official_name(gaz, raw):
    """`Residencial` alone scores 100 against `Unidad Residencial El Coliseo`.

    A fuzzy hit is only plausible when the mention explains at least half of the
    official name's tokens, or when both have the same shape and differ only in
    spelling.
    """
    assert gaz.detect(raw).barrio is None


def test_single_word_typo_is_still_tolerated(gaz):
    """The plausibility filter must not kill same-shape spelling differences."""
    assert gaz.detect("Barrio Belalcazer").barrio == "Belalcázar"


def test_fuzzy_match_still_prefers_the_shorter_name_at_equal_coverage(gaz):
    det = gaz.detect("Cl. 44a #6N-42, La Flora")
    assert det.barrio == "La Flora"


def test_segment_fuzzy_match_tolerates_a_typo(gaz):
    det = gaz.detect("Cl. 16 #35-57, Santa Elen")
    assert det.barrio == "Santa Elena"


def test_multiline_input_with_label(gaz):
    det = gaz.detect("Calle 44 # 3-4\nBarrio La Flora\nCali")
    assert det.barrio == "Urbanización La Flora" or det.barrio == "La Flora"
    assert parse_address(det.cleaned_text).parse_ok


# ---------------------------------------------------------------------------
# Detection - in-core tail (rule c)
# ---------------------------------------------------------------------------


def test_trailing_mention_after_plate_with_dash(gaz):
    det = gaz.detect("Calle 3ra A oeste # 44-06 - SILOE")
    assert det.barrio == "Siloé"
    assert det.comuna == 20


def test_trailing_mention_after_plate_without_separator(gaz):
    det = gaz.detect("Calle 3ra A oeste # 44-06 SILOE")
    assert det.barrio == "Siloé"
    assert parse_address(det.cleaned_text).canonical().startswith("CL 3 A OESTE")


@pytest.mark.parametrize(
    "raw",
    [
        "Carrera 5 # 12-30",
        "Calle 5 # 38-25",
        "Cra 9 # 9-15",
        "CL 18 # 100-04",
        "cl 13 # 39-27 edif Jane",
    ],
)
def test_no_detection_inside_a_plain_address_core(gaz, raw):
    det = gaz.detect(raw)
    assert (det.barrio, det.vereda, det.comuna, det.corregimiento) == (None, None, None, None), det
    assert det.cleaned_text == raw


def test_generic_short_name_does_not_fire_from_a_longer_tail(gaz):
    """`Sucre` / `Lili` are real barrios but far too generic for a loose tail."""
    det = gaz.detect("Calle 5 # 38-25 casa de lili")
    assert det.barrio is None


def test_multiword_name_fires_from_the_tail(gaz):
    det = gaz.detect("Calle 44 # 3-4 urbanizacion la flora")
    assert det.barrio == "La Flora"


# ---------------------------------------------------------------------------
# Detection - degenerate input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "-", "0", "N/A", "###", "...", 0, float("nan")])
def test_no_detection_on_empty_or_junk(gaz, raw):
    det = gaz.detect(raw)
    assert (det.barrio, det.vereda, det.comuna, det.corregimiento) == (None, None, None, None)
    assert isinstance(det.cleaned_text, str)


def test_very_long_input_is_handled(gaz):
    det = gaz.detect("Calle 5 # 38-25, " + "x" * 5000)
    assert det.barrio is None


def test_detect_is_deterministic(gaz):
    a = gaz.detect("Cl. 16 #35-57, Santa Elena, Cali")
    b = gaz.detect("Cl. 16 #35-57, Santa Elena, Cali")
    assert (a.barrio, a.comuna, a.cleaned_text) == (b.barrio, b.comuna, b.cleaned_text)


# ---------------------------------------------------------------------------
# Precedence / inference
# ---------------------------------------------------------------------------


def test_barrio_implies_its_comuna(gaz):
    det = gaz.detect("Barrio Siloé")
    assert (det.barrio, det.comuna) == ("Siloé", 20)
    assert "comuna_from_barrio" in det.notes


def test_explicit_comuna_agreeing_with_barrio_has_no_conflict_note(gaz):
    det = gaz.detect("Barrio Siloé, comuna 20")
    assert det.comuna == 20
    assert "comuna_barrio_conflict" not in det.notes


def test_explicit_comuna_conflicting_with_barrio_is_flagged(gaz):
    det = gaz.detect("Barrio Siloé, comuna 15")
    assert det.barrio == "Siloé"
    assert det.comuna == 15
    assert "comuna_barrio_conflict" in det.notes


def test_explicit_comuna_wins_over_inference(gaz):
    det = gaz.detect("comuna 15, barrio Siloé")
    assert det.comuna == 15


# ---------------------------------------------------------------------------
# cleaned_text
# ---------------------------------------------------------------------------


def test_cleaned_text_parses_where_raw_did_not_danubio(gaz):
    raw = "Danubio\nCarrera 77-No 1C-140\nUrbanización Danubio"
    assert not parse_address(raw).parse_ok
    det = gaz.detect(raw)
    assert det.barrio is None  # no `Danubio` barrio exists in the official layer
    assert parse_address(det.cleaned_text).parse_ok
    assert parse_address(det.cleaned_text).canonical() == "KR 77 # 1 C - 140"


def test_cleaned_text_parses_after_removing_a_leading_label(gaz):
    raw = "Barrio Siloé calle 1 # 2-3"
    assert not parse_address(raw).parse_ok
    det = gaz.detect(raw)
    assert parse_address(det.cleaned_text).parse_ok


def test_cleaned_text_is_unchanged_when_nothing_is_detected(gaz):
    raw = "Cra. 98f #98-66 apto 301"
    det = gaz.detect(raw)
    assert det.cleaned_text == raw


def test_cleaned_text_keeps_the_address_core_intact(gaz):
    det = gaz.detect("Barrio Siloé, Calle 3 A oeste # 44-06")
    assert "44-06" in det.cleaned_text
    assert "3 A oeste" in det.cleaned_text
    assert "SILO" not in det.cleaned_text.upper()


def test_cleaned_text_drops_the_comuna_label(gaz):
    det = gaz.detect("Cra. 34 #14c-48, Comuna 10, Cali, Valle del Cauca")
    assert det.comuna == 10
    assert "COMUNA" not in det.cleaned_text.upper()
    assert parse_address(det.cleaned_text).canonical() == "KR 34 # 14 C - 48"


# ---------------------------------------------------------------------------
# locate
# ---------------------------------------------------------------------------


def test_locate_known_urban_coordinate(gaz):
    got = gaz.locate(SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    assert got["barrio"] == "San Fernando Nuevo"
    assert got["id_barrio"] == "1906"
    assert got["comuna"] == 19
    assert got["vereda"] is None


def test_locate_rural_coordinate_returns_vereda_and_corregimiento(gaz):
    entry = gaz.lookup("vereda", "El Pinar")
    pt = entry.geometry.representative_point()
    got = gaz.locate(pt.x, pt.y)
    assert got["vereda"] == "El Pinar"
    assert got["corregimiento"] == "La Castilla"
    assert got["barrio"] is None


@pytest.mark.parametrize("lon, lat", [(0.0, 0.0), (-74.0, 4.6), (float("nan"), 3.4), (None, None)])
def test_locate_outside_the_city_is_all_none(gaz, lon, lat):
    got = gaz.locate(lon, lat)
    assert got["barrio"] is None and got["vereda"] is None and got["comuna"] is None


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


def test_contains_inside(gaz):
    assert gaz.contains("barrio", "San Fernando Nuevo", SAN_FERNANDO_LON, SAN_FERNANDO_LAT)


def test_contains_rejects_a_point_in_another_barrio(gaz):
    assert not gaz.contains("barrio", "Siloé", SAN_FERNANDO_LON, SAN_FERNANDO_LAT)


def test_contains_at_a_boundary_needs_the_buffer(gaz):
    """A point ~100 m outside the polygon: rejected bare, accepted with 200 m."""
    entry = gaz.lookup("barrio", "San Fernando Nuevo")
    outside = entry.geometry.buffer(100.0 / 111_000.0).exterior.coords[0]
    lon, lat = float(outside[0]), float(outside[1])
    assert not gaz.contains("barrio", "San Fernando Nuevo", lon, lat)
    assert gaz.contains("barrio", "San Fernando Nuevo", lon, lat, buffer_m=200.0)


def test_contains_buffer_is_not_unbounded(gaz):
    assert not gaz.contains("barrio", "Siloé", SAN_FERNANDO_LON, SAN_FERNANDO_LAT, buffer_m=200.0)


def test_contains_unknown_name_is_false(gaz):
    assert not gaz.contains("barrio", "Zzzqqq", SAN_FERNANDO_LON, SAN_FERNANDO_LAT)


def test_contains_unknown_kind_is_false(gaz):
    assert not gaz.contains("planeta", "Siloé", SAN_FERNANDO_LON, SAN_FERNANDO_LAT)


@pytest.mark.parametrize("lon, lat", [(float("nan"), 3.4), (None, 3.4), (-76.5, None)])
def test_contains_with_missing_coordinates_is_false(gaz, lon, lat):
    assert not gaz.contains("barrio", "Siloé", lon, lat)


def test_contains_comuna_by_number_or_name(gaz):
    assert gaz.contains("comuna", 19, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    assert gaz.contains("comuna", "Comuna 19", SAN_FERNANDO_LON, SAN_FERNANDO_LAT)


# ---------------------------------------------------------------------------
# Geographic gate (CLI helper, no model needed)
# ---------------------------------------------------------------------------


def _cand(score, doc, lon, lat):
    return Candidate(score=score, doc=doc, lon=lon, lat=lat)


def test_gate_keeps_candidates_inside_the_detected_barrio(gaz):
    det = gaz.detect("Barrio San Fernando Nuevo")
    inside = _cand(0.9, 1, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    kept, label = gate_candidates(gaz, det, [inside])
    assert kept == [inside]
    assert label == "barrio San Fernando Nuevo"


def test_gate_drops_candidates_outside_the_detected_barrio(gaz):
    det = gaz.detect("Barrio San Fernando Nuevo")
    far = _cand(0.95, 2, -76.60, 3.44)  # Siloé side of town
    near = _cand(0.80, 1, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    kept, label = gate_candidates(gaz, det, [far, near])
    assert kept == [near]
    assert label == "barrio San Fernando Nuevo"


def test_gate_returns_empty_when_every_candidate_is_outside(gaz):
    det = gaz.detect("Barrio San Fernando Nuevo")
    kept, label = gate_candidates(gaz, det, [_cand(0.95, 2, -76.60, 3.44)])
    assert kept == []
    assert label == "barrio San Fernando Nuevo"


def test_gate_uses_the_comuna_when_no_barrio_was_detected(gaz):
    det = gaz.detect("comuna 19")
    inside = _cand(0.9, 1, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    outside = _cand(0.9, 2, -76.60, 3.44)
    kept, label = gate_candidates(gaz, det, [inside, outside])
    assert kept == [inside]
    assert label == "comuna 19"


def test_gate_uses_the_corregimiento_for_rural_detections(gaz):
    det = gaz.detect("Corregimiento Pance")
    pance = gaz.lookup("corregimiento", "Pance").geometry.representative_point()
    inside = _cand(0.9, 1, pance.x, pance.y)
    outside = _cand(0.9, 2, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    kept, label = gate_candidates(gaz, det, [inside, outside])
    assert kept == [inside]
    assert label == "corregimiento Pance"


def test_gate_is_a_no_op_without_any_detection(gaz):
    det = gaz.detect("Carrera 5 # 12-30")
    cands = [_cand(0.9, 1, -76.60, 3.44), _cand(0.5, 2, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)]
    kept, label = gate_candidates(gaz, det, cands)
    assert kept == cands
    assert label is None


def test_gate_is_a_no_op_on_an_empty_candidate_list(gaz):
    det = gaz.detect("Barrio Siloé")
    kept, label = gate_candidates(gaz, det, [])
    assert kept == []
    assert label == "barrio Siloé"


def test_gate_keeps_candidates_with_missing_coordinates(gaz):
    """A candidate without a centroid cannot be proven outside, so it survives."""
    det = gaz.detect("Barrio Siloé")
    blind = _cand(0.9, 1, float("nan"), float("nan"))
    kept, _ = gate_candidates(gaz, det, [blind])
    assert kept == [blind]


def test_gate_preserves_input_order(gaz):
    det = gaz.detect("comuna 19")
    a = _cand(0.9, 1, SAN_FERNANDO_LON, SAN_FERNANDO_LAT)
    b = _cand(0.8, 2, SAN_FERNANDO_LON + 0.0001, SAN_FERNANDO_LAT)
    kept, _ = gate_candidates(gaz, det, [a, b])
    assert kept == [a, b]


def test_gate_escalates_to_the_comuna_when_the_barrio_empties(gaz):
    """Colloquial barrio extents miss the official polygon; the comuna still holds."""
    det = gaz.detect("Barrio Santa Elena")
    assert det.comuna == 10
    # a point in Cristóbal Colón: same comuna, neighbouring barrio
    outside_barrio = _cand(0.9, 1, -76.524825, 3.421087)
    kept, label = gate_candidates(gaz, det, [outside_barrio])
    assert kept == []
    kept, label = gate_candidates(gaz, det, [outside_barrio], escalate=True)
    assert kept == [outside_barrio]
    assert label == "comuna 10"


def test_gate_escalation_still_rejects_another_comuna(gaz):
    det = gaz.detect("Barrio Santa Elena")
    far = _cand(0.9, 1, -76.60, 3.44)
    kept, label = gate_candidates(gaz, det, [far], escalate=True)
    assert kept == []
    assert label == "comuna 10"


def test_gate_barrio_buffer_is_configurable(gaz):
    det = gaz.detect("Barrio Santa Elena")
    outside_barrio = _cand(0.9, 1, -76.524825, 3.421087)  # ~465 m out
    assert gate_candidates(gaz, det, [outside_barrio], barrio_buffer_m=200.0)[0] == []
    assert gate_candidates(gaz, det, [outside_barrio], barrio_buffer_m=1000.0)[0] == [outside_barrio]


def test_gate_barrio_buffer_admits_a_near_miss(gaz):
    """200 m of slack absorbs centroid/geometry noise at barrio borders."""
    entry = gaz.lookup("barrio", "San Fernando Nuevo")
    outside = entry.geometry.buffer(100.0 / 111_000.0).exterior.coords[0]
    det = gaz.detect("Barrio San Fernando Nuevo")
    kept, _ = gate_candidates(gaz, det, [_cand(0.9, 1, float(outside[0]), float(outside[1]))])
    assert len(kept) == 1
