"""Address-only CLI for the Cali cadastral address normalizer.

All the batching, scoring, gating and IO logic lives in
``cali_address.service`` so this CLI and the HTTP service in
``cali_address.api`` cannot drift apart. This file is argument parsing plus
stdout/stderr formatting, nothing else; read ``src/cali_address/service.py``
for the status contract, the strictness rules and the geographic gate.

Statuses
--------
OK                  parsed, matched above the tuned confidence cutoff, inside the
                    detected place (when one was detected), and the matched
                    cadastral address agrees structurally with the input
NO_PARSEABLE        the address grammar could not parse the text (the detected
                    place columns are still filled)
SIN_MATCH           no cadastral record can be assigned; ``motivo`` says why

Place gazetteer and geographic gate
-----------------------------------
``cali_address.gazetteer`` reads the IDESC basemaps under ``basemaps/`` and does
two things per row:

1. It detects place mentions (``Barrio Siloé``, ``vereda la reforma``,
   ``COMUNA 15``, a trailing ``, Navarro``) and strips them, so the address
   grammar sees clean nomenclature. ``Barrio Siloé calle 1 # 2-3`` does not parse
   at all; ``calle 1 # 2-3`` does.
2. It turns the detection into a hard spatial constraint. Cali repeats its
   nomenclature across the city, so the reranker regularly prefers a
   text-identical predio in the wrong neighbourhood. Candidates whose centroid
   lies outside the detected barrio (buffered ``--barrio-buffer``, default
   1000 m) - or, when only a comuna or corregimiento is known, outside that
   polygon (buffered ``--zone-buffer``, default 300 m) - are dropped before the
   best candidate is chosen. With ``--gate-escalate`` (default on) an empty barrio
   gate falls back to the comuna polygon; if nothing survives, the row becomes
   ``SIN_MATCH`` with reason ``candidatos fuera de <kind> <name>`` instead of
   silently keeping a wrong predio.

Rows without a detected place are scored exactly as before. Pass
``--no-gazetteer`` to disable both steps.

Output zone columns
-------------------
``barrio_vereda`` and ``comuna_corregimiento`` hold ONE normalized value each
(official IDESC names, comunas as ``Comuna 19``). On OK rows they are the polygons
the matched predio falls in; on other rows they come from the place detected in
the text, so rural and unparseable addresses still get a normalized zone.

Output address columns
----------------------
Exactly two address columns are written: ``direccion_entrada`` (or the kept
source column when ``--keep-columns`` includes it) and ``direccion_normalizada``.
``fuente_normalizacion`` says where the normalized value comes from:
``catastro`` (the matched cadastral record, only on OK rows), ``reglas`` (the
deterministic grammar output, parseable but unmatched rows) or empty.

Input columns
-------------
``--column`` is optional: the address column is otherwise scored against a
Spanish/English synonym set and auto-picked only when the guess is unambiguous.
If two columns are equally plausible the CLI prints the ranked candidates and
exits non-zero instead of silently normalizing the wrong field. ``--header`` is
also optional: when omitted the header row is detected (the ArcGIS Excel exports
prefix the real header with 2-5 metadata rows).

What the gate costs, measured
-----------------------------
On ``stickers`` (3 697 rows, which carry an inspector GPS coordinate), the text
cleaning alone raises OK from 2 421 to 2 458. The 200 m barrio gate then withholds
138 of those rows, and 86 % of the matches it discarded sat within 100 m of the
recorded coordinate: the barrio names in that file come from a reverse geocoder
whose colloquial extents do not line up with the official IDESC polygons
(the barrio named in the text agrees with the barrio polygon the matched predio
sits in only 77 % of the time, while the comuna agrees 92 % of the time). The gate is
therefore tunable: ``--barrio-buffer 1000`` keeps 2 436 OK rows at the same 3.1 %
share of >500 m errors, and ``--gate-escalate`` falls back to the comuna instead of
withholding the row. Those are the defaults; ``--barrio-buffer 200
--no-gate-escalate`` restores the strict behaviour.

Usage
-----
    python scripts/normalizar.py "Calle 5 # 38-25" "Cra. 98f #98-66"
    python scripts/normalizar.py --input context/stickers.xlsx --column direccion \
        --sheet stickers --header 5 --output salida.xlsx
    python scripts/normalizar.py --input direcciones.txt --json --no-gazetteer
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cali_address.gazetteer import BARRIO_BUFFER_M, ZONE_BUFFER_M  # noqa: E402
from cali_address.inference import AddressNormalizer  # noqa: E402
from cali_address.service import (  # noqa: E402
    CLI_BARRIO_BUFFER_M,
    DEFAULT_MIN_STRUCT,
    AddressColumnError,
    KeepColumnsError,
    TableFormatError,
    attach_source_columns,
    load_gazetteer,
    normalize_strict,
    read_input_file,
    write_output,
)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
BASEMAPS_DIR = os.path.join(PROJECT_ROOT, "basemaps")


def _print_address_column_error(exc: AddressColumnError) -> None:
    """Explain which column to pass instead of dumping a traceback."""
    print(f"error: {exc.message}", file=sys.stderr)
    if exc.candidates:
        print("candidatas (mayor puntaje primero):", file=sys.stderr)
        for item in exc.candidates:
            print(f"  --column {item['column']!r}  (puntaje {item['score']:.1f})", file=sys.stderr)
    if exc.available:
        print(f"columnas disponibles: {exc.available}", file=sys.stderr)
    print("use --column para indicar la columna de direccion", file=sys.stderr)


def _plate_tolerance_type(value: str) -> int:
    """argparse type for ``--plate-tolerance``: an int clamped to [0, 2].

    Larger values silently widen false OK matches, so the CLI rejects them
    the same way the API's ``Tunables.plate_tolerance`` (``ge=0, le=2``) does.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if not 0 <= parsed <= 2:
        raise argparse.ArgumentTypeError(
            f"--plate-tolerance must be between 0 and 2 (got {parsed})"
        )
    return parsed


def _ambiguity_delta_type(value: str) -> float:
    """argparse type for ``--ambiguity-delta``: a float in [0, 0.2] (the API's ``ge=0, le=0.2``)."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if not 0.0 <= parsed <= 0.2:
        raise argparse.ArgumentTypeError(f"--ambiguity-delta must be between 0 and 0.2 (got {parsed})")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Strict, address-only normalizer for Cali cadastral addresses.")
    ap.add_argument("addresses", nargs="*", help="one or more raw addresses")
    ap.add_argument("--input", "-i", help="xlsx/csv/geojson/zip/txt file with addresses")
    ap.add_argument("--column", "-c", help="address column name (auto-detected when omitted)")
    ap.add_argument("--sheet", help="sheet name (xlsx)")
    ap.add_argument("--header", type=int, default=None,
                    help="0-based header row (xlsx/csv); detected automatically when omitted")
    ap.add_argument("--output", "-o", help="write results to .csv or .xlsx (xlsx gets sheets normalizado / revisar / resumen)")
    ap.add_argument("--keep-columns", help="comma-separated input columns to carry into the output, or 'all'")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--min-struct", type=float, default=DEFAULT_MIN_STRUCT, help="minimum structural agreement (0-1) to accept a match")
    ap.add_argument("--plate-tolerance", type=_plate_tolerance_type, default=0,
                    help="accept a matched plate that differs by at most N, 0-2 (default 0 = exact plate)")
    ap.add_argument("--ambiguity-delta", type=_ambiguity_delta_type, default=0.02,
                    help="abstain (SIN_MATCH 'ambiguo') when the score gap to the best rule-passing candidate in "
                         "another manzana is below this, 0-0.2 (default 0.02; 0 = off)")
    ap.add_argument("--basemaps", default=BASEMAPS_DIR, help="directory with the IDESC basemaps (default: basemaps/)")
    ap.add_argument("--no-gazetteer", action="store_true", help="disable place detection and the geographic gate")
    ap.add_argument("--barrio-buffer", type=float, default=CLI_BARRIO_BUFFER_M,
                    help=f"metres of slack around a detected barrio (default {BARRIO_BUFFER_M:.0f})")
    ap.add_argument("--zone-buffer", type=float, default=ZONE_BUFFER_M,
                    help=f"metres of slack around a detected comuna / corregimiento (default {ZONE_BUFFER_M:.0f})")
    ap.add_argument("--gate-escalate", action=argparse.BooleanOptionalAction, default=True,
                    help="fall back to the comuna/corregimiento polygon when no candidate is inside the barrio "
                         "(default: on; --no-gate-escalate withholds the row instead)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    source, address_column = None, None
    if args.input:
        try:
            raws, source, address_column = read_input_file(
                args.input, args.column, args.sheet, args.header
            )
        except AddressColumnError as exc:
            _print_address_column_error(exc)
            return 2
        except TableFormatError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.addresses:
        raws = args.addresses
    else:
        ap.error("provide addresses as arguments or --input FILE")

    gazetteer = None if args.no_gazetteer else load_gazetteer(
        args.basemaps, warn=lambda message: print(message, file=sys.stderr)
    )
    normalizer = AddressNormalizer(ARTIFACTS_DIR, device=args.device)
    stats: dict = {}
    result = normalize_strict(
        normalizer, raws, min_struct=args.min_struct, plate_tolerance=args.plate_tolerance,
        gazetteer=gazetteer, stats=stats, barrio_buffer_m=args.barrio_buffer,
        zone_buffer_m=args.zone_buffer, gate_escalate=args.gate_escalate,
        ambiguity_delta=args.ambiguity_delta,
    )
    try:
        result = attach_source_columns(result, source, args.keep_columns, address_column)
    except KeepColumnsError as exc:
        sys.exit(str(exc))

    if args.output:
        write_output(result, args.output)
        print(f"written {len(result)} rows -> {args.output}", file=sys.stderr)

    if args.json:
        records = result.astype(object).where(pd.notna(result), None).to_dict(orient="records")
        print(json.dumps(records, ensure_ascii=False, indent=2))
    elif not args.output:
        with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 40):
            print(result.to_string(index=False))

    counts = result["estado"].value_counts().to_dict()
    print(f"summary: {counts}", file=sys.stderr)
    if gazetteer is not None:
        print(f"gazetteer: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
