"""Edge-case oriented tests for the rule-based Cali address parser.

Coverage groups: empty/None/whitespace, numeric and textual junk, glued tokens,
missing `#`, quadrants, BIS, ordinal noise, multi-address strings, unicode and
accents, very long complements, legacy `K`/`C`/`A`/`D`/`T` prefixes, plate
padding, idempotency and non-raising behaviour on adversarial input.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cali_address.parser import (  # noqa: E402
    canonical,
    canonicalize,
    match_key,
    parse_address,
    parse_many,
)


# --------------------------------------------------------------------------
# 1. Empty / null / whitespace / junk -> parse_ok must be False, never raise
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "\t\n",
        "16758424",          # the junk value present in acciones_candidatos
        "0",
        "-",
        "#",
        "# -",
        "...",
        ";;;",
        "123 456 789",       # digits only, no via type
        "SIN ESPECIFICAR",
        "NO REPORTA",
        "N/A",
        "El predio queda al lado de la tienda",
    ],
)
def test_unparseable_inputs_are_flagged_not_raised(value):
    result = parse_address(value)
    assert result.parse_ok is False
    assert canonical(result) == ""
    assert isinstance(result.notes, list) and result.notes


def test_none_and_empty_notes():
    assert "null_input" in parse_address(None).notes
    assert "empty_input" in parse_address("   ").notes


# --------------------------------------------------------------------------
# 2. Canonical happy paths (cadastral round-trip)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("KR 26 H 1 # 73 - 10", "KR 26 H 1 # 73 - 10"),
        ("CL 12 # 48 BIS - 31", "CL 12 # 48 BIS - 31"),
        ("AV 15 OESTE # 9 OESTE - 137", "AV 15 OESTE # 9 OESTE - 137"),
        ("CL 25 NORTE # AV 6 - 30", "CL 25 NORTE # AV 6 - 30"),
        ("KR 28 D 3 # 72 L - 04", "KR 28 D 3 # 72 L - 04"),
        ("CL 72 T 1 # 26 G 11 - 70", "CL 72 T 1 # 26 G 11 - 70"),
        ("CL 12 A # 52 - 60 SO 1 PQ 19", "CL 12 A # 52 - 60 SO 1 PQ 19"),
    ],
)
def test_cadastral_round_trip(raw, expected):
    assert canonicalize(raw) == expected


# --------------------------------------------------------------------------
# 3. Legacy one-letter via prefixes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("K 47B  # 55 B  - Q2 T", "KR 47 B # 55 B - Q2 T"),
        ("C 9 W  # 0  -", "CL 9 W # 0 -"),
        ("A 8 C BIS  # 23 B  - 17", "AV 8 C BIS # 23 B - 17"),
        ("D 26 H 4  # T 93  - 11", "DG 26 H 4 # TV 93 - 11"),
        ("T 28 F  # 70  - 00", "TV 28 F # 70 - 00"),
    ],
)
def test_legacy_prefixes_are_expanded(raw, expected):
    assert canonicalize(raw) == expected


# --------------------------------------------------------------------------
# 4. Glued tokens
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CL12A # 56-04", "CL 12 A # 56 - 04"),
        ("Carrera1#9-80", "KR 1 # 9 - 80"),
        ("CRA62C#6A-26", "KR 62 C # 6 A - 26"),
        ("Cl.55b # 47-55", "CL 55 B # 47 - 55"),
        ("KR38BIS#5-09", "KR 38 BIS # 5 - 09"),
    ],
)
def test_glued_tokens(raw, expected):
    assert canonicalize(raw) == expected


# --------------------------------------------------------------------------
# 5. Missing '#'
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,via,cross,plate",
    [
        ("Calle 9c 49 141", ("CL", "9", "C"), "49", "141"),
        ("KR 13 60 34 38", ("KR", "13", None), "60", "34"),
        ("Calle 10 no. 42a 02", ("CL", "10", None), "42", "02"),
        ("CARRERA 62 6 26", ("KR", "62", None), "6", "26"),
    ],
)
def test_missing_hash_is_recovered(raw, via, cross, plate):
    p = parse_address(raw)
    assert p.parse_ok
    assert (p.via_type, p.via_number, p.via_letters) == via
    assert p.cross_number == cross
    assert p.plate is not None and f"{int(p.plate):02d}" == plate


def test_missing_hash_note():
    assert "implicit_hash" in parse_address("Calle 9c 49 141").notes
    assert "explicit_hash" in parse_address("CL 9 C # 49 - 141").notes


# --------------------------------------------------------------------------
# 6. 'No.' / 'Nro' / 'N°' / 'num' variants
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "Calle 10 No. 42 - 02",
        "Calle 10 No 42 - 02",
        "Calle 10 N. 42 - 02",
        "Calle 10 N° 42 - 02",
        "Calle 10 Nro 42 - 02",
        "Calle 10 Nro. 42 - 02",
        "Calle 10 numero 42 - 02",
        "Calle 10 num 42-02",
        "Calle 10 # 42 - 02",
    ],
)
def test_number_word_variants(raw):
    assert canonicalize(raw) == "CL 10 # 42 - 02"


# --------------------------------------------------------------------------
# 7. Quadrants (including typos) and BIS
# --------------------------------------------------------------------------

def test_quadrant_spelled_out():
    p = parse_address("Avenida 6 Oeste # 22 - 14")
    assert (p.via_type, p.via_number, p.via_quadrant) == ("AV", "6", "OESTE")


def test_quadrant_typo_is_repaired():
    p = parse_address("Av 6 Oeate #22-14")
    assert p.via_quadrant == "OESTE"
    assert canonical(p) == "AV 6 OESTE # 22 - 14"


def test_quadrant_two_letter_abbreviation():
    assert parse_address("AV 4 OE # 10 - 20").via_quadrant == "OESTE"
    assert parse_address("CL 25 NTE # 6 - 30").via_quadrant == "NORTE"


def test_single_letter_after_number_is_a_street_letter_not_a_quadrant():
    # The cadastre writes quadrants in full (NORTE/OESTE); `E`/`N`/`W` in this
    # position are street letters (`KR 41 E`, `C 60 N`, `K 73 B # 2 W`).
    p = parse_address("KR 41 E # 30 - 74")
    assert p.via_letters == "E"
    assert p.via_quadrant is None
    assert canonical(p) == "KR 41 E # 30 - 74"


def test_cross_quadrant():
    p = parse_address("CL 73 # 2 B NORTE - 39")
    assert (p.cross_number, p.cross_letters, p.cross_quadrant) == ("2", "B", "NORTE")


@pytest.mark.parametrize("raw", ["CL 12 # 48 BIS - 31", "CL 12 # 48BIS-31", "CL 12 # 48 bis - 31"])
def test_bis_in_cross(raw):
    p = parse_address(raw)
    assert p.cross_bis is True
    assert canonical(p) == "CL 12 # 48 BIS - 31"


def test_bis_in_via():
    p = parse_address("K 32 A BIS  # 42 C - 145")
    assert p.via_bis is True
    assert canonical(p) == "KR 32 A BIS # 42 C - 145"


# --------------------------------------------------------------------------
# 8. Ordinal noise
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Calle 3ra # 5 - 10", "CL 3 # 5 - 10"),
        ("Carrera 5ta # 8 - 20", "KR 5 # 8 - 20"),
        ("Calle 1ra # 2da - 30", "CL 1 # 2 - 30"),
    ],
)
def test_ordinal_noise_is_stripped(raw, expected):
    assert canonicalize(raw) == expected


# --------------------------------------------------------------------------
# 9. Multiple addresses and separators
# --------------------------------------------------------------------------

def test_multiple_addresses_takes_the_first():
    p = parse_address("CALLE 5 B2 # 38-91; CRA 37 #4A BIS 49")
    assert p.parse_ok
    assert p.via_type == "CL" and p.via_number == "5"
    assert "multiple_addresses_first_taken" in p.notes


def test_underscore_and_en_dash_become_plate_separator():
    assert canonicalize("KR 1C 3 # 64 A _41") == canonicalize("KR 1C 3 # 64 A - 41")
    assert canonicalize("CL 10 # 20 – 30") == "CL 10 # 20 - 30"


# --------------------------------------------------------------------------
# 10. Trailing city / neighbourhood noise
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "Cl. 55b # 47-55, Navarro, Cali, Valle del Cauca",
        "Cl. 55b # 47-55, Cali",
        "Cl. 55b # 47-55 Cali Valle del Cauca",
        "CL 55 B # 47 - 55 - SILOE",
    ],
)
def test_trailing_noise_is_dropped(raw):
    p = parse_address(raw)
    assert (p.via_type, p.via_number, p.via_letters) == ("CL", "55", "B")
    assert p.cross_number == "47"


def test_city_noise_note_is_recorded():
    notes = parse_address("Cl. 55b # 47-55, Cali, Valle del Cauca").notes
    assert "dropped_city_noise" in notes or "dropped_trailing_segment" in notes


# --------------------------------------------------------------------------
# 11. Unicode / accents / case
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "Calle 10 # 42 - 02",
        "calle 10 # 42 - 02",
        "CALLE 10 # 42 - 02",
        "Cálle 10 # 42 - 02",
        "Cålle 10 # 42 - 02",
    ],
)
def test_case_and_accent_insensitivity(raw):
    assert canonicalize(raw) == "CL 10 # 42 - 02"


def test_non_latin_characters_do_not_crash():
    for raw in ["Дирекция", "住所 12", "\U0001f600 CL 10 # 42 - 02"]:
        parse_address(raw)  # must not raise


# --------------------------------------------------------------------------
# 12. Complements, including very long ones
# --------------------------------------------------------------------------

def test_complement_abbreviations_are_canonicalized():
    p = parse_address("CL 10 # 42 - 02 APARTAMENTO 501 TORRE 3")
    assert p.complement == "AP 501 TO 3"


def test_very_long_complement_is_preserved():
    tail = " ".join(f"TOKEN{i}" for i in range(80))
    p = parse_address(f"CL 10 # 42 - 02 {tail}")
    assert p.parse_ok
    assert p.complement is not None
    assert p.complement.count("TOKEN") == 80
    assert canonical(p, with_complement=False) == "CL 10 # 42 - 02"


def test_complement_can_be_excluded_from_canonical():
    p = parse_address("CL 10 # 42 - 02 AP 501")
    assert canonical(p, with_complement=True) == "CL 10 # 42 - 02 AP 501"
    assert canonical(p, with_complement=False) == "CL 10 # 42 - 02"


def test_glued_complement_token_is_not_split():
    p = parse_address("C 26  # 68 B  - 83 14C")
    assert p.plate == "83"
    assert p.complement == "14C"


# --------------------------------------------------------------------------
# 13. Plate handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,plate", [("CL 10 # 42 - 2", "02"), ("CL 10 # 42 - 02", "02"),
                                       ("CL 10 # 42 - 137", "137"), ("CL 10 # 42 - 0", "00")])
def test_plate_is_zero_padded_to_two_digits(raw, plate):
    assert canonicalize(raw).endswith(f"- {plate}")


def test_missing_plate_keeps_the_separator():
    p = parse_address("KR 40 B # 10 -")
    assert p.plate is None and p.had_plate_sep is True
    assert canonical(p) == "KR 40 B # 10 -"


def test_missing_cross_and_plate():
    assert canonicalize("C 183 3 C  #  -") == "CL 183 3 C # -"


# --------------------------------------------------------------------------
# 14. Styles, match_key, idempotency
# --------------------------------------------------------------------------

def test_glued_style_matches_idesc_output():
    p = parse_address("Cra. 98f #98-66")
    assert canonical(p, style="glued") == "KR 98F # 98 - 66"
    assert canonical(p, style="spaced") == "KR 98 F # 98 - 66"


def test_match_key_ignores_whitespace_and_punctuation():
    assert match_key("KR 98 F # 98 - 66") == match_key("KR98F#98-66")
    assert match_key("CL 12 A") == "CL12A"


@pytest.mark.parametrize(
    "raw",
    [
        "Cra. 98f #98-66",
        "CL12A # 56-04",
        "Calle 9c 49 141",
        "K 47B  # 55 B  - Q2 T",
        "Av 6 Oeate #22-14",
        "CL 12 # 48 BIS - 31",
        "Cl. 55b # 47-55, Navarro, Cali, Valle del Cauca",
    ],
)
def test_canonical_form_is_idempotent(raw):
    once = canonicalize(raw)
    assert once, raw
    assert canonicalize(once) == once


# --------------------------------------------------------------------------
# 15. Robustness fuzzing: never raise, never return a partial canonical string
# --------------------------------------------------------------------------

def test_fuzz_never_raises():
    rng = random.Random(42)
    alphabet = "ABCKLR#-_ 0123456789.,;ñá°\t"
    for _ in range(3000):
        n = rng.randint(0, 40)
        raw = "".join(rng.choice(alphabet) for _ in range(n))
        p = parse_address(raw)
        assert isinstance(p.parse_ok, bool)
        out = canonical(p)
        assert isinstance(out, str)
        if p.parse_ok:
            assert p.via_type and p.via_number


def test_parse_many_handles_mixed_batch():
    values = ["CL 10 # 42 - 02", None, "", "junk", "KR 1 # 2 - 3"]
    results = parse_many(values)
    assert [r.parse_ok for r in results] == [True, False, False, False, True]


def test_extremely_long_input_is_bounded():
    raw = "CL 10 # 42 - 02 " + ("X" * 5000)
    p = parse_address(raw)
    assert p.parse_ok is True
    assert p.via_number == "10"


# --------------------------------------------------------------------------
# 16. Segmentation heuristics learned from the real messy datasets
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # cross with a secondary number and a zero-padded plate as last token
        ("KR 38 BIS # 5B2 09", "KR 38 BIS # 5 B 2 - 09"),
        ("Cra 37 # 5B1 37", "KR 37 # 5 B 1 - 37"),
        # bare number run: first number is the cross, second the plate
        ("KR 13 60 34 38", "KR 13 # 60 - 34 38"),
        # explicit plate dash but no '#'
        ("Catrera 36 5b3-65", "KR 36 # 5 B 3 - 65"),
    ],
)
def test_plate_segmentation_heuristics(raw, expected):
    assert canonicalize(raw) == expected


def test_via_type_typo_is_repaired():
    assert parse_address("Catrera 36 # 5 - 65").via_type == "KR"
    assert parse_address("Cllle 36 # 5 - 65").via_type == "CL"


def test_glued_cardinal_suffix_on_street_letter():
    p = parse_address("Av 5 An #23 Dn 68")
    assert (p.via_letters, p.via_quadrant) == ("A", "NORTE")
    assert (p.cross_letters, p.cross_quadrant) == ("D", "NORTE")
    assert canonical(p) == "AV 5 A NORTE # 23 D NORTE - 68"


def test_quadrant_word_inside_implicit_cross():
    assert canonicalize("CRA 100 1B OESTE 110 TORRE 7 APTO 402") == \
        "KR 100 # 1 B OESTE - 110 TO 7 AP 402"


# --------------------------------------------------------------------------
# 17. Dash without '#': the number after it is the cross, not the plate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["Calle 5 - 38", "CL 5 - 38", "Calle 5-38"])
def test_dash_without_hash_is_cross_not_plate(raw):
    # Without an explicit '#', `Calle 5 - 38` means via 5, cross 38, no plate -
    # not via 5, no cross, plate 38.
    p = parse_address(raw)
    assert p.parse_ok
    assert (p.via_type, p.via_number) == ("CL", "5")
    assert p.cross_number == "38"
    assert p.plate is None
    assert canonical(p) == "CL 5 # 38"


def test_dash_without_hash_via_cross_plate_still_works():
    # Regression: when something already sits between the via and the dash
    # (a cross candidate on the left, e.g. `36 5B3`), the right side of the
    # dash stays the plate, not the cross.
    p = parse_address("Catrera 36 5b3-65")
    assert (p.via_type, p.via_number) == ("KR", "36")
    assert (p.cross_number, p.cross_letters, p.cross_suffix) == ("5", "B", "3")
    assert p.plate == "65"
    assert canonical(p) == "KR 36 # 5 B 3 - 65"
