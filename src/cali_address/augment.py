"""Stochastic corruption model that turns a canonical cadastral address into the
kind of free text people actually type.

Every operation was chosen after reading the five evaluation workbooks in
``context/``; the comment next to each one quotes a real example.
"""

from __future__ import annotations

import random
import re

__all__ = ["corrupt", "corrupt_many", "OPERATIONS"]

VIA_EXPANSIONS = {
    "CL": ["CALLE", "CLL", "CLLE", "Cl.", "Calle", "CL", "C", "calle", "Cll", "CALLE"],
    "KR": ["CARRERA", "CRA", "CR", "KRA", "Cra.", "Carrera", "KR", "K", "carrera", "Cra", "CRA."],
    "AV": ["AVENIDA", "AV", "Av.", "Avenida", "AVDA", "Av", "avenida"],
    "DG": ["DIAGONAL", "DG", "Diagonal", "DIAG"],
    "TV": ["TRANSVERSAL", "TV", "Transversal", "TRV"],
    "AC": ["AVENIDA CALLE", "AC", "AV CALLE"],
    "AK": ["AVENIDA CARRERA", "AK", "AV CARRERA"],
}
QUADRANT_ABBREV = {
    "NORTE": ["NORTE", "Norte", "N", "NTE", "Nte.", "norte", "n"],
    "OESTE": ["OESTE", "Oeste", "O", "OE", "W", "oeste", "Oes"],
    "SUR": ["SUR", "Sur", "S", "sur"],
    "ESTE": ["ESTE", "Este", "E", "este"],
}
HASH_VARIANTS = ["#", "No.", "No", "N°", "Nro", "Nro.", "numero", "num", "N.", "#", "#", ""]
PLATE_SEPARATORS = [" - ", "-", " ", "_", " – ", "  -", "- "]
CITY_SUFFIXES = [
    ", Cali, Valle del Cauca", ", Cali", " Cali", ", Cali, Valle del Cauca, Colombia",
    " CALI", ", Valle del Cauca",
]
BARRIO_SUFFIXES = [
    ", El Guabal", ", San Judas Tadeo", ", Mariano Ramos", ", Pampa Linda", " san bosco",
    ", Navarro", " - SILOE", ", Santa Anita", " El Poblado", ", Alfonso Lopez",
]
COMPLEMENTS = [
    "APTO 301", "APT 201", "AP 502", "TORRE 2", "TO 3", "CASA 4", "LOCAL 1", "PISO 3",
    "BLOQUE 3C APTO 302", "CONJUNTO RESIDENCIAL LOS ALAMOS", "INT 1", "MANZANA 14 LOTE 17",
    "EDIF CANELO", "BQ 01", "zona social", "LC 2",
]
ORDINALS = {"1": "1ra", "2": "2da", "3": "3ra", "4": "4ta", "5": "5ta", "6": "6ta",
            "7": "7ma", "8": "8va", "9": "9na"}

OPERATIONS = (
    "expand_via_type", "expand_cross_type", "hash_variant", "plate_separator",
    "glue_letters", "glue_all", "lowercase", "city_suffix", "barrio_suffix",
    "add_complement", "drop_complement", "typo", "ordinal", "quadrant_abbrev",
    "unpad_plate", "drop_plate", "whitespace_jitter", "duplicate_address",
)

_SEG_RE = re.compile(
    r"^(?P<via>[A-Z]{2,4}(?:\s+[A-Z]{2,4})?)\s+(?P<vianum>\d+)(?P<viarest>[^#]*)"
    r"(?:#(?P<cross>[^-]*)(?:-\s*(?P<plate>\d+)?\s*(?P<comp>.*))?)?$"
)


def _typo(text: str, rng: random.Random, n: int = 1) -> str:
    chars = list(text)
    for _ in range(n):
        if len(chars) < 3:
            break
        i = rng.randrange(len(chars))
        op = rng.choice(("swap", "delete", "duplicate", "replace"))
        if op == "swap" and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "delete":
            del chars[i]
        elif op == "duplicate":
            chars.insert(i, chars[i])
        else:
            chars[i] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(chars)


def corrupt(
    canon: str, rng: random.Random, intensity: float = 1.0, op_scale: dict | None = None
) -> str:
    """Return one noisy variant of a canonical cadastral address.

    ``intensity`` scales every operation probability (0 -> identity).
    ``op_scale`` optionally multiplies the probability of individual operations
    (keys are names from :data:`OPERATIONS`); ``None`` reproduces the original
    behaviour bit for bit. See :mod:`cali_address.realistic_noise`.
    """
    scale = op_scale or {}
    text = canon.strip()
    if not text:
        return text

    def p(prob: float, op: str | None = None) -> bool:
        return rng.random() < min(prob * intensity * scale.get(op, 1.0), 0.98)

    # Split into via / cross / plate / complement on the cadastral markers.
    via_part, sep, tail = text.partition("#")
    via_part = via_part.strip()
    cross_part, dash, plate_part = (tail.partition(" - ") if " - " in tail else (tail, "", ""))
    cross_part = cross_part.strip()
    plate_part = plate_part.strip()
    plate_num, _, comp = plate_part.partition(" ")

    # -- via type: `KR` -> `CARRERA` / `Cra.` / `K` ... ("Cra. 61a # 9-16")
    via_tokens = via_part.split()
    if via_tokens and p(0.75, "expand_via_type"):
        head = via_tokens[0]
        two = " ".join(via_tokens[:2])
        if two in VIA_EXPANSIONS:
            via_tokens[:2] = [rng.choice(VIA_EXPANSIONS[two])]
        elif head in VIA_EXPANSIONS:
            via_tokens[0] = rng.choice(VIA_EXPANSIONS[head])
        via_part = " ".join(via_tokens)

    # -- cross type (rare, e.g. `# AV 6`)
    cross_tokens = cross_part.split()
    if cross_tokens and cross_tokens[0] in VIA_EXPANSIONS and p(0.4, "expand_cross_type"):
        cross_tokens[0] = rng.choice(VIA_EXPANSIONS[cross_tokens[0]])
        cross_part = " ".join(cross_tokens)

    # -- quadrant abbreviations ("Cra5 norte # 33n -01", "Av. 4 Nte. #37 Norte-48")
    if p(0.55, "quadrant_abbrev"):
        for full, variants in QUADRANT_ABBREV.items():
            repl = rng.choice(variants)
            via_part = re.sub(rf"\b{full}\b", repl, via_part)
            cross_part = re.sub(rf"\b{full}\b", repl, cross_part)

    # -- ordinal noise ("CL 1 RA D OESTE # 100 BIS 76")
    if p(0.06, "ordinal"):
        m = re.match(r"^(\S+)\s+(\d)\b", via_part)
        if m and m.group(2) in ORDINALS:
            via_part = via_part.replace(f" {m.group(2)}", f" {ORDINALS[m.group(2)]}", 1)

    # -- plate: drop the zero padding ("KR 43 # 10-50" vs "- 05")
    if plate_num.isdigit() and p(0.3, "unpad_plate"):
        plate_num = str(int(plate_num))

    # -- complement: drop it (most user input has no unit) or swap it
    if comp and p(0.55, "drop_complement"):
        comp = ""
    elif not comp and p(0.22, "add_complement"):
        comp = rng.choice(COMPLEMENTS)
    elif comp and p(0.2, "add_complement"):
        comp = rng.choice(COMPLEMENTS)

    # -- reassemble with a random hash marker and plate separator
    hash_marker = rng.choice(HASH_VARIANTS) if p(0.65, "hash_variant") else "#"
    plate_sep = rng.choice(PLATE_SEPARATORS) if p(0.6, "plate_separator") else " - "
    out = via_part
    if cross_part or plate_num:
        out += f" {hash_marker} " if hash_marker else " "
        out += cross_part
        if plate_num and not p(0.05, "drop_plate"):           # occasionally drop the plate
            out += f"{plate_sep}{plate_num}"
    if comp:
        out += f" {comp}"
    out = re.sub(r"\s+", " ", out).strip()

    # -- glue tokens ("Calle12a#56-04", "Carrera1#9-80", "Cra 94C1#2A-42")
    if p(0.35, "glue_letters"):
        out = re.sub(r"(?<=\d)\s+(?=[A-Za-z]\b)", "", out)   # `12 A` -> `12A`
    if p(0.25, "glue_all"):
        out = re.sub(r"\s*#\s*", "#", out)
    if p(0.12, "glue_all"):
        out = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", out)     # `Calle 12` -> `Calle12`

    # -- trailing administrative noise
    if p(0.18, "city_suffix"):
        out += rng.choice(CITY_SUFFIXES)
    if p(0.1, "barrio_suffix"):
        out += rng.choice(BARRIO_SUFFIXES)

    # -- Fasecolda duplicates the address inside the same cell
    if p(0.04, "duplicate_address"):
        out = f"{out} {out}"

    # -- case
    r = rng.random()
    if r < 0.30 * intensity * scale.get("lowercase", 1.0):
        out = out.lower()
    elif r < 0.42 * intensity * scale.get("lowercase", 1.0):
        out = out.title()

    # -- typos last, so they can hit any part of the string
    if p(0.22, "typo"):
        out = _typo(out, rng, n=1 if rng.random() < 0.75 else 2)
    if p(0.1, "whitespace_jitter"):
        out = out.replace(" ", "  ", 1)

    return out.strip()


def corrupt_many(
    canon: str, n: int, rng: random.Random, op_scale: dict | None = None
) -> list[str]:
    """``n`` distinct-ish variants; the first one is always mildly corrupted."""
    out = [corrupt(canon, rng, intensity=0.45, op_scale=op_scale)]
    for _ in range(max(n - 1, 0)):
        out.append(corrupt(canon, rng, intensity=1.0, op_scale=op_scale))
    return out
