"""Command line: ``cali-address {setup,normalize,web,inspect,formats}`` (or ``python -m cali_address ...``).

Exit codes: 0 ok, 1 aborted (``--on-error raise`` hit a failing row, or out of memory), 2 usage / mapping /
config error, 3 I/O error (missing or corrupt input, unreachable database, unwritable output, missing model
artifacts), 4 the run finished but EVERY row was ERROR (e.g. a broken model; the output and summary are still
written). User errors print one ``error:`` line (plus actionable hints) to stderr, never a traceback.

The pre-existing ``scripts/normalizar.py`` keeps working unchanged; this CLI adds
any-format input, column mapping, chunked streaming and a run summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Callable

import pandas as pd

from .paths import ArtifactsMissingError
from .io import (
    ColumnMapping,
    DatasetError,
    DatasetIOError,
    MemorySink,
    MissingDependencyError,
    UsageError,
    describe_reader_formats,
    describe_writer_formats,
    infer_format,
    load_config,
    merge_options,
    open_sink,
    read_table,
    resolve_config_paths,
)
from .io.readers import _canonical_format, redact_source

log = logging.getLogger(__name__)

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_IO, EXIT_ALL_ROWS_FAILED = 0, 1, 2, 3, 4
PROG = "python -m cali_address"


def _project_defaults() -> dict:
    """Option defaults (lazy: importing the service pulls in torch)."""
    from .gazetteer import ZONE_BUFFER_M
    from .paths import resolve_artifacts_dir
    from .service import CLI_BARRIO_BUFFER_M, DEFAULT_BASEMAPS_DIR, DEFAULT_MIN_STRUCT

    return {
        "chunk_size": 20_000, "parts_sep": " ", "on_error": "mark", "gazetteer": True, "gate_escalate": True,
        "min_struct": DEFAULT_MIN_STRUCT, "plate_tolerance": 0, "ambiguity_delta": 0.02,
        "barrio_buffer": CLI_BARRIO_BUFFER_M, "zone_buffer": ZONE_BUFFER_M,
        "artifacts_dir": resolve_artifacts_dir(), "basemaps": DEFAULT_BASEMAPS_DIR,
    }


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _ranged_int(name: str, lo: int, hi: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
        if not lo <= parsed <= hi:
            raise argparse.ArgumentTypeError(f"--{name} must be between {lo} and {hi} (got {parsed})")
        return parsed

    return parse


def _ranged_float(name: str, lo: float, hi: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
        if not lo <= parsed <= hi:
            raise argparse.ArgumentTypeError(f"--{name} must be between {lo} and {hi} (got {parsed})")
        return parsed

    return parse


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {parsed})")
    return parsed


def _delimiter(value: str) -> str:
    return {"tab": "\t", "\\t": "\t", "comma": ",", "semicolon": ";", "pipe": "|"}.get(value.lower(), value)


def _add_source_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", help="input format (default: inferred from the extension / URL scheme); see 'formats'")
    p.add_argument("--sheet", help="sheet name (xlsx/xls); default: the first sheet")
    p.add_argument("--header-row", "--header", dest="header_row", type=int, default=None,
                   help="0-based header row; detected automatically when omitted")
    p.add_argument("--delimiter", type=_delimiter, help="CSV delimiter (default: sniffed among , ; TAB |)")
    p.add_argument("--encoding", help="text encoding (default: BOM / UTF-8 / cp1252 detected)")
    p.add_argument("--table", help="table or view name for SQL sources ([schema.]name) or layer name for GPKG")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description="Normalize Cali addresses from any tabular dataset.")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{setup,normalize,web,inspect,formats}")

    n = sub.add_parser("normalize", help="normalize the address column of a dataset",
                       description="Normalize addresses from a file, URL or database into a file.")
    n.add_argument("input", nargs="?", help="file, http(s) URL or SQLAlchemy URL (or set 'input' in --config)")
    n.add_argument("-o", "--output", help="output file: .csv .tsv .xlsx .parquet .json .jsonl .geojson")
    n.add_argument("--output-format", help="force the output format")
    _add_source_options(n)
    n.add_argument("--address-col", "--column", "-c", dest="address_col",
                   help="address column (auto-detected when omitted)")
    n.add_argument("--address-parts", help="comma separated columns to join into one address, in order")
    n.add_argument("--parts-sep", help="separator used to join --address-parts (default: a space)")
    n.add_argument("--id-col", help="id column (always carried to the output)")
    n.add_argument("--municipality-col",
                   help="municipality column: non-empty values other than Cali become FUERA_DE_AREA")
    n.add_argument("--lat-col", help="latitude column (informational only, never used for matching)")
    n.add_argument("--lon-col", help="longitude column (informational only, never used for matching)")
    n.add_argument("--keep-columns", help="'all' (default), 'none' or a comma separated list of input columns")
    n.add_argument("--chunk-size", type=_positive_int, help="rows per chunk (default 20000)")
    n.add_argument("--on-error", choices=["mark", "raise"], help="'mark' (default) turns failing rows into ERROR rows")
    n.add_argument("--artifacts-dir", help="model artifacts directory (default: artifacts/)")
    n.add_argument("--basemaps", help="IDESC basemaps directory (default: basemaps/)")
    n.add_argument("--threshold", type=_ranged_float("threshold", 0.0, 1.0),
                   help="override the tuned confidence cutoff")
    n.add_argument("--min-struct", type=_ranged_float("min-struct", 0.0, 1.0),
                   help="minimum structural agreement (0-1) to accept a match")
    n.add_argument("--plate-tolerance", type=_ranged_int("plate-tolerance", 0, 2),
                   help="accept a plate that differs by at most N, 0-2 (default 0)")
    n.add_argument("--ambiguity-delta", type=_ranged_float("ambiguity-delta", 0.0, 0.2),
                   help="abstain when the gap to a competitor in another manzana is below this, 0-0.2 (default 0.02)")
    n.add_argument("--barrio-buffer", type=float, help="metres of slack around a detected barrio")
    n.add_argument("--zone-buffer", type=float, help="metres of slack around a detected comuna / corregimiento")
    n.add_argument("--gate-escalate", action=argparse.BooleanOptionalAction, default=None,
                   help="fall back to the comuna polygon when no candidate is inside the barrio (default: on)")
    n.add_argument("--gazetteer", action=argparse.BooleanOptionalAction, default=None,
                   help="place detection and geographic gate (default: on; --no-gazetteer disables)")
    n.add_argument("--soft-rules", help="comma separated rule names whose violations are soft")
    n.add_argument("--max-soft", type=_ranged_int("max-soft", 0, 20), help="soft violations an OK row may carry")
    n.add_argument("--gate-fallback", action=argparse.BooleanOptionalAction, default=None,
                   help="keep text-ranked candidates when the geographic gate would leave none")
    n.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    n.add_argument("--dry-run", type=_positive_int, metavar="N",
                   help="process only the first N rows, print them and write nothing")
    n.add_argument("--summary-json", help="write the run summary to this JSON file")
    n.add_argument("--config", help="TOML file with the same options; command line flags override it")

    i = sub.add_parser("inspect", help="show format, columns, guessed address column and a preview",
                       description="Read a dataset WITHOUT normalizing it.")
    i.add_argument("input", help="file, http(s) URL or SQLAlchemy URL")
    _add_source_options(i)
    i.add_argument("--json", action="store_true", help="machine readable output")

    sub.add_parser("formats", help="list the supported input and output formats")

    st = sub.add_parser("setup", help="make the repository runnable (regenerates catastro_emb.pt) and self-check it",
                        description="Check the model artifacts, regenerate catastro_emb.pt when it is the only missing "
                                    "file, then normalize 3 invented addresses to prove everything works.")
    st.add_argument("--artifacts-dir", help="model artifacts directory (default: artifacts/)")
    st.add_argument("--force", action="store_true", help="regenerate catastro_emb.pt even if it exists")
    st.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    st.add_argument("--batch-size", type=_positive_int, default=4096, help="documents per embedding batch (default 4096)")

    w = sub.add_parser("web", help="start the browser upload page (needs the 'api' extra)",
                       description="Serve the HTTP API and its upload page on this machine.")
    w.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1: this machine only)")
    w.add_argument("--port", type=_ranged_int("port", 1, 65535), default=8000, help="TCP port (default 8000)")
    w.add_argument("--artifacts-dir", help="model artifacts directory (default: artifacts/)")
    w.add_argument("--device", help="cuda or cpu (default: the API default, cpu)")
    w.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return ap


# ---------------------------------------------------------------------------
# error reporting
# ---------------------------------------------------------------------------
def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _report(exc: Exception) -> int:
    from .tables import AddressColumnError

    if isinstance(exc, AddressColumnError):
        _err("several columns could be the address; choose one." if exc.candidates else exc.message)
        for item in exc.candidates:
            print(f"  --address-col {item['column']!r}  (score {item['score']:.1f})", file=sys.stderr)
        if exc.available:
            print(f"available columns: {', '.join(exc.available)}", file=sys.stderr)
        print("use --address-col NAME (or --address-parts A,B,C) to choose the address", file=sys.stderr)
        return EXIT_USAGE
    _err(str(exc))
    return EXIT_USAGE if isinstance(exc, UsageError) else EXIT_IO


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
#: Output extension for each input format when ``-o`` is omitted (formats with no writer map to a close one).
_DEFAULT_OUT_EXT = {
    "csv": ".csv", "tsv": ".tsv", "txt": ".csv", "xlsx": ".xlsx", "xls": ".xlsx", "parquet": ".parquet",
    "json": ".json", "jsonl": ".jsonl", "geojson": ".geojson", "shp": ".geojson", "gpkg": ".geojson",
}


def default_output_path(source: str, fmt: str) -> str:
    """``<input stem>_normalizado<ext>`` next to a local input (same format; xls -> xlsx, txt -> csv,
    shp/zip/gpkg -> geojson). The suffix guarantees the result never equals the input path."""
    source = os.fspath(source)
    stem = os.path.splitext(source)[0]
    return f"{stem}_normalizado{_DEFAULT_OUT_EXT.get(fmt, '.csv')}"


def _cli_options(args: argparse.Namespace) -> dict:
    keys = ["input", "output", "output_format", "format", "sheet", "header_row", "delimiter", "encoding", "table",
            "address_col", "address_parts", "parts_sep", "id_col", "municipality_col", "lat_col",
            "lon_col", "keep_columns", "chunk_size", "on_error", "artifacts_dir", "basemaps", "threshold",
            "min_struct", "plate_tolerance", "ambiguity_delta", "barrio_buffer", "zone_buffer", "gate_escalate",
            "gazetteer", "soft_rules", "max_soft", "gate_fallback", "device", "dry_run", "summary_json"]
    return {k: getattr(args, k, None) for k in keys}


def _read_kwargs(opts: dict) -> dict:
    return {
        "fmt": opts.get("format"), "encoding": opts.get("encoding"), "delimiter": opts.get("delimiter"),
        "sheet": opts.get("sheet"), "header_row": opts.get("header_row"), "table": opts.get("table"),
    }


def _default_normalizer(artifacts_dir: str, device: str | None):
    from .paths import require_artifacts

    require_artifacts(artifacts_dir)  # friendly error before the slow torch import
    from .inference import AddressNormalizer

    return AddressNormalizer(artifacts_dir, device=device)


def _default_gazetteer(basemaps_dir: str, warn=None):
    from .service import load_gazetteer

    return load_gazetteer(basemaps_dir, warn=warn)


def _print_summary(summary: dict, elapsed_note: str = "") -> None:
    counts = " ".join(f"{k}={v}" for k, v in summary["by_estado"].items())
    print(
        f"processed {summary['rows']} rows in {summary['seconds']:.1f}s "
        f"({summary['rows_per_sec']:.0f} rows/s){elapsed_note}",
        file=sys.stderr,
    )
    print(f"summary: {counts}", file=sys.stderr)
    if summary.get("by_nivel_precision"):
        print(f"nivel_precision: {summary['by_nivel_precision']}", file=sys.stderr)
    for line in summary.get("error_samples", []):
        print(f"  ERROR {line}", file=sys.stderr)
    if summary.get("gazetteer"):
        print(f"gazetteer: {summary['gazetteer']}", file=sys.stderr)


def _cmd_normalize(args, normalizer_factory, gazetteer_loader) -> int:
    from .io.pipeline import Tunables, normalize_dataset  # lazy: pulls in the model stack

    file_options = {}
    if args.config:  # relative paths in the file are relative to the file, not to the CWD
        file_options = resolve_config_paths(load_config(args.config), os.path.dirname(os.path.abspath(args.config)))
    if file_options.get("no_gazetteer") is True and "gazetteer" not in file_options:
        file_options["gazetteer"] = False
    file_options.pop("no_gazetteer", None)
    opts = merge_options(_project_defaults(), file_options, _cli_options(args))

    source = opts.get("input")
    if not source:
        _err("no input given: pass INPUT (a file, URL or SQLAlchemy URL) or set 'input' in --config")
        return EXIT_USAGE
    dry_run = opts.get("dry_run")
    if not opts.get("output") and not dry_run:
        if "://" in str(source):
            _err("-o/--output is required for URL and database inputs (or use --dry-run N to preview without writing)")
            return EXIT_USAGE
        fmt = _canonical_format(opts["format"]) if opts.get("format") else infer_format(source)
        opts["output"] = default_output_path(source, fmt)

    mapping = ColumnMapping.from_options(
        address_col=opts.get("address_col"), address_parts=opts.get("address_parts"),
        parts_sep=opts.get("parts_sep"), id_col=opts.get("id_col"), municipality_col=opts.get("municipality_col"),
        lat_col=opts.get("lat_col"), lon_col=opts.get("lon_col"), keep_columns=opts.get("keep_columns"),
    )
    soft_rules = opts.get("soft_rules")
    if isinstance(soft_rules, str):
        soft_rules = [s.strip() for s in soft_rules.split(",") if s.strip()]
    tunables = Tunables(
        min_struct=opts["min_struct"], plate_tolerance=opts["plate_tolerance"],
        ambiguity_delta=opts["ambiguity_delta"], barrio_buffer_m=opts["barrio_buffer"],
        zone_buffer_m=opts["zone_buffer"], gate_escalate=opts["gate_escalate"],
        soft_rules=tuple(soft_rules) if soft_rules is not None else None, max_soft=opts.get("max_soft"),
        gate_fallback=opts.get("gate_fallback"), threshold=opts.get("threshold"),
    )

    # Everything that can fail on the user's side happens BEFORE the (slow) model is loaded.
    reader = read_table(source, chunk_size=opts["chunk_size"], **_read_kwargs(opts))
    sink = None
    published = False
    try:
        mapping.resolve(reader.columns)
        sink = MemorySink() if dry_run else open_sink(opts["output"], opts.get("output_format"))
        for warning in reader.warnings:
            print(f"warning: {warning}", file=sys.stderr)

        try:
            normalizer = normalizer_factory(opts["artifacts_dir"], opts.get("device"))
        except ArtifactsMissingError as exc:
            _err(str(exc))
            return EXIT_IO
        except (FileNotFoundError, OSError) as exc:
            _err(f"cannot load the model artifacts from {opts['artifacts_dir']!r}: {exc}. Use --artifacts-dir.")
            return EXIT_IO
        if tunables.threshold is not None:
            normalizer.threshold = float(tunables.threshold)
        gazetteer = None
        if opts["gazetteer"]:
            gazetteer = gazetteer_loader(opts["basemaps"], warn=lambda message: print(message, file=sys.stderr))

        tty = sys.stderr.isatty()

        def progress(done: int, chunk_no: int) -> None:
            if tty:
                print(f"\r  {done} rows", end="", file=sys.stderr, flush=True)

        try:
            summary = normalize_dataset(
                reader, mapping, sink, normalizer=normalizer, gazetteer=gazetteer, chunk_size=opts["chunk_size"],
                on_error=opts["on_error"], progress=progress, tunables=tunables, limit=dry_run,
            )
        except DatasetError:
            raise
        except MemoryError:
            _err("out of memory; nothing was written. Retry with a smaller --chunk-size.")
            return EXIT_FAILED
        except Exception as exc:
            if opts["on_error"] == "raise":  # the caller asked for fail-fast: report it, without a traceback
                _err(f"aborted by --on-error raise: {type(exc).__name__}: {exc}")
                return EXIT_FAILED
            raise
        reader.close()  # release the input before publishing: -o may be the input path itself (Windows locks it)
        sink.close()
        published = True
    finally:
        try:
            if sink is not None and not published:
                try:
                    sink.abort()  # no partial output on any failure path, KeyboardInterrupt included
                except Exception as abort_exc:  # never mask the error that got us here
                    log.debug("sink.abort() failed: %s", abort_exc)
        finally:
            reader.close()
    if tty:
        print(file=sys.stderr)

    if dry_run:
        print("dry run: nothing was written", file=sys.stderr)
        with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
            print(sink.frame.to_string(index=False))
    else:
        print(f"written {summary['rows']} rows -> {opts['output']}", file=sys.stderr)
    _print_summary(summary)
    if opts.get("summary_json"):
        target = opts["summary_json"]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
        except OSError as exc:
            _err(f"cannot write the summary to {target}: {exc}")
            return EXIT_IO
    if summary["rows"] > 0 and summary["error"] == summary["rows"]:
        _err(f"every row failed ({summary['error']} of {summary['rows']} are ERROR); the model or its inputs are "
             "probably broken. The output and the summary were still written.")
        return EXIT_ALL_ROWS_FAILED
    return EXIT_OK


# ---------------------------------------------------------------------------
# setup / web
# ---------------------------------------------------------------------------
def _cmd_setup(args, normalizer_factory) -> int:
    from .paths import ENV_ARTIFACTS_DIR, REQUIRED_ARTIFACTS, check_artifacts, resolve_artifacts_dir

    art = resolve_artifacts_dir(args.artifacts_dir)
    missing, missing_optional = check_artifacts(art)
    missing_names = {os.path.basename(p) for p in missing}
    print(f"artifacts directory: {art}")
    for name in REQUIRED_ARTIFACTS:
        print(f"  [{'MISSING' if name in missing_names else 'ok'}] {name}")
    for path in missing_optional:
        print(f"  [absent] {os.path.basename(path)} (optional)")

    hard_missing = [n for n in ("model.pt", "catastro_docs.parquet") if n in missing_names]
    if hard_missing:
        _err(f"cannot set up: {', '.join(hard_missing)} not found in {art}.")
        print("  These files are tracked in git: run 'git pull' (or 'git checkout -- artifacts') in the repository.\n"
              "  Otherwise see docs/model-artifacts.md, or point --artifacts-dir / "
              f"{ENV_ARTIFACTS_DIR} at a directory that has them.", file=sys.stderr)
        return EXIT_IO

    from .bootstrap import regenerate_embeddings, self_check
    from .train import resolve_device

    device = resolve_device(args.device)
    if args.device and args.device.startswith("cuda") and device == "cpu":
        print("note: CUDA is not available, using the CPU")
    if "catastro_emb.pt" in missing_names or args.force:
        print(f"regenerating catastro_emb.pt (device: {device}). Expected time: seconds on a GPU, "
              "a few minutes (about 3-10) on a CPU.", flush=True)
        last = [-1]

        def progress(done: int, total: int) -> None:
            pct = done * 100 // total
            if done == total or pct // 10 > last[0] // 10:
                last[0] = pct
                print(f"  embedding {done}/{total} ({pct}%)", flush=True)

        try:
            info = regenerate_embeddings(art, device=device, batch_size=args.batch_size, progress=progress)
        except OSError as exc:
            _err(f"cannot write catastro_emb.pt in {art}: {exc}")
            return EXIT_IO
        except Exception as exc:
            _err(f"could not regenerate catastro_emb.pt: {type(exc).__name__}: {exc}")
            return EXIT_FAILED
        print(f"wrote {info['path']} ({info['rows']} rows x {info['dim']}) in {info['seconds']:.1f}s")
    else:
        print("Already set up: nothing to regenerate.")

    try:
        result = self_check(normalizer_factory, art, device)
    except Exception as exc:
        _err(f"Setup FAILED: the self-check could not normalize the sample addresses ({type(exc).__name__}: {exc}). "
             "If catastro_emb.pt is stale or corrupt, run 'cali-address setup --force'.")
        return EXIT_FAILED
    print(f"Setup OK: 3 sample addresses normalized, {result['rows_per_sec']:.0f} rows/s on {device}. "
          "Next: cali-address normalize YOUR_FILE.xlsx")
    return EXIT_OK


def _missing_web_extras() -> list[str]:
    missing = []
    for names in (("fastapi",), ("uvicorn",), ("python_multipart", "multipart")):  # multipart: pre-0.0.13 name
        for name in names:
            try:
                __import__(name)
                break
            except ImportError:
                continue
        else:
            missing.append(names[0])
    return missing


def _schedule_browser(url: str, delay: float = 1.5) -> None:
    timer = threading.Timer(delay, webbrowser.open, args=(url,))
    timer.daemon = True
    timer.start()


def _cmd_web(args) -> int:
    from .paths import require_artifacts, resolve_artifacts_dir

    if _missing_web_extras():
        _err('the web page needs the "api" extra: pip install -e ".[api]"')
        return EXIT_IO
    art = resolve_artifacts_dir(args.artifacts_dir)
    try:
        require_artifacts(art)
    except ArtifactsMissingError as exc:
        _err(str(exc))
        print("run 'cali-address setup' first", file=sys.stderr)
        return EXIT_IO
    probe = socket.socket(socket.AF_INET6 if ":" in args.host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((args.host, args.port))
    except OSError as exc:
        _err(f"cannot listen on {args.host}:{args.port} ({exc.strerror or exc}); is another server using it? "
             "Try --port 8001.")
        return EXIT_IO
    finally:
        probe.close()

    import uvicorn

    os.environ["ARTIFACTS_DIR"] = art  # the API's own variable (api/main.py artifacts_dir())
    if args.device:
        os.environ["NORMALIZER_DEVICE"] = args.device
    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}"
    print(f"Starting the web page at {url}  (the model loads in the background; Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        _schedule_browser(url)
    uvicorn.run("cali_address.api.main:app", host=args.host, port=args.port)
    return EXIT_OK


# ---------------------------------------------------------------------------
# inspect / formats
# ---------------------------------------------------------------------------
_COUNT_LIMIT_BYTES = 100 * 1024 * 1024


def _count_rows(source: str, opts: dict, fmt: str) -> int | None:
    """Row count when it is cheap (local file under 100 MB, or Parquet metadata)."""
    if fmt in ("sql", "gpkg", "xls") or "://" in source:
        return None
    try:
        if fmt == "parquet":
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(source).metadata.num_rows)
        if os.path.getsize(source) > _COUNT_LIMIT_BYTES:
            return None
        kwargs = {**_read_kwargs(opts), "fmt": fmt}
        with read_table(source, chunk_size=100_000, **kwargs) as chunks:
            return sum(len(c) for c in chunks)
    except (DatasetError, OSError):
        return None


def _cmd_inspect(args) -> int:
    from .tables import AddressColumnError, detect_address_column, resolve_address_column

    opts = _cli_options(args)
    source = args.input
    fmt = opts["format"] or infer_format(source)
    reader = read_table(source, chunk_size=1000, **{**_read_kwargs(opts), "fmt": fmt})
    with reader:
        first = next(reader, None)
        columns = reader.columns
        preview = (first.head(5) if first is not None else pd.DataFrame(columns=columns))
        info = {
            "source": redact_source(source), "format": reader.format, "delimiter": reader.delimiter, "encoding": reader.encoding,
            "header_row": reader.header_row, "sheets": reader.sheet_names, "columns": list(columns),
            "warnings": list(reader.warnings),
        }
    _, ranked = detect_address_column(columns)
    guessed, status, candidates = None, "none", [c["column"] for c in ranked if c["score"] >= 70.0]
    try:
        guessed, status = resolve_address_column(pd.DataFrame(columns=columns), None), "unique"
    except AddressColumnError as exc:
        status = "ambiguous" if exc.candidates else "none"
    info.update({
        "guessed_address_column": guessed, "address_column_status": status,
        "address_candidates": candidates, "row_count": _count_rows(source, opts, reader.format),
        "preview": preview.astype(object).where(pd.notna(preview), None).to_dict(orient="records"),
    })
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK

    def show(label: str, value) -> None:
        if value not in (None, [], ""):
            print(f"{label}: {value}")

    show("source", info["source"])
    detail = ", ".join(f"{k} {v!r}" for k, v in (("delimiter", info["delimiter"]), ("encoding", info["encoding"]),
                                                  ("header row", info["header_row"])) if v is not None)
    print(f"format: {info['format']}" + (f" ({detail})" if detail else ""))
    show("sheets", ", ".join(info["sheets"] or []))
    print(f"columns ({len(columns)}): {', '.join(columns)}")
    if status == "unique":
        print(f"guessed address column: {guessed}")
    elif status == "ambiguous":
        print(f"guessed address column: ambiguous between {', '.join(candidates)} - pass --address-col")
    else:
        print("guessed address column: none - pass --address-col or --address-parts")
    count = info["row_count"]
    print(f"rows: {count if count is not None else 'unknown (large or non-file source)'}")
    for warning in info["warnings"]:
        print(f"warning: {warning}")
    print("first rows:")
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
        print(preview.to_string(index=False))
    return EXIT_OK


def _cmd_formats() -> int:
    def table(title: str, rows: list[dict]) -> None:
        print(title)
        for row in rows:
            exts = " ".join(row["extensions"]) or "(explicit --format / URL)"
            print(f"  {row['format']:<8} {exts:<22} {row['note']}")

    table("input formats:", describe_reader_formats())
    print()
    table("output formats:", describe_writer_formats())
    print("\nSQL sources: pass a SQLAlchemy URL (sqlite:///f.db, postgresql://u:p@host/db) with --table (a table or a view).")
    return EXIT_OK


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None, *, normalizer_factory=None, gazetteer_loader=None) -> int:
    """Entry point. ``normalizer_factory(artifacts_dir, device)`` and ``gazetteer_loader(basemaps, warn=)``
    are injectable so tests (and embedders) can run without the model."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "normalize":
            return _cmd_normalize(args, normalizer_factory or _default_normalizer,
                                  gazetteer_loader or _default_gazetteer)
        if args.command == "setup":
            return _cmd_setup(args, normalizer_factory or _default_normalizer)
        if args.command == "web":
            return _cmd_web(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        return _cmd_formats()
    except Exception as exc:
        from .tables import AddressColumnError

        if isinstance(exc, (DatasetError, AddressColumnError)):
            return _report(exc)
        raise
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
