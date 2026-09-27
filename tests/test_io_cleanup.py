"""Every path out of a run releases the reader, never leaves partial output, and reports fatal errors."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from dataset_helpers import ADDR_OK, ADDR_RAISES, ExplodingStub, CADASTRE, exploding_stub, stub, write_csv

from cali_address import cli as cli_module
from cali_address.io import ColumnMapping, MemorySink, SourceReadError, normalize_dataset
from cali_address.io.readers import TableChunks, read_table
from cali_address.io.writers import Sink


class RecordingSink(Sink):
    def __init__(self, fail_on_write: BaseException | None = None) -> None:
        self.fail_on_write = fail_on_write
        self.writes = self.closes = self.aborts = 0

    def write(self, df):
        self.writes += 1
        if self.fail_on_write is not None:
            raise self.fail_on_write

    def close(self):
        self.closes += 1

    def abort(self):
        self.aborts += 1


def _chunks(frames, columns=("direccion",)):
    """A TableChunks over ``frames`` that counts how many times its closer ran."""
    state = {"closed": 0}
    reader = TableChunks("chunks", list(columns), iter(frames), closers=[lambda: state.__setitem__("closed", state["closed"] + 1)])
    return reader, state


def _frames(n_chunks=3):
    return [pd.DataFrame({"direccion": [ADDR_OK, ADDR_OK]}) for _ in range(n_chunks)]


def _normalize(source, *, normalizer=None, sink=None, **kw):
    return normalize_dataset(
        source, ColumnMapping(address_col="direccion"), sink or MemorySink(),
        normalizer=normalizer or stub(), chunk_size=2, **kw,
    )


class RaisingStub(ExplodingStub):
    """Raises ``exc`` for a batch that holds ADDR_RAISES."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__(CADASTRE)
        self.exc = exc

    def score_components(self, queries, k=20):
        if any(ADDR_RAISES in q for q in queries):
            raise self.exc
        return super(ExplodingStub, self).score_components(queries, k=k)


def _poisoned():
    return [pd.DataFrame({"direccion": [ADDR_OK, ADDR_OK]}), pd.DataFrame({"direccion": [ADDR_RAISES, ADDR_OK]})]


# ---------------------------------------------------------------------------
# pipeline: reader is closed on every path
# ---------------------------------------------------------------------------
def test_reader_closed_on_success():
    reader, state = _chunks(_frames())
    _normalize(reader)
    assert state["closed"] == 1


def test_reader_closed_when_limit_stops_early():
    reader, state = _chunks(_frames(5))
    _normalize(reader, limit=3)
    assert state["closed"] == 1


def test_reader_closed_on_normalizer_error_with_on_error_raise():
    reader, state = _chunks(_poisoned())
    with pytest.raises(RuntimeError):
        _normalize(reader, normalizer=exploding_stub(), on_error="raise")
    assert state["closed"] == 1


def test_reader_closed_when_sink_write_fails():
    reader, state = _chunks(_frames())
    with pytest.raises(OSError):
        _normalize(reader, sink=RecordingSink(fail_on_write=OSError("disk full")))
    assert state["closed"] == 1


def test_reader_closed_on_mapping_error():
    reader, state = _chunks(_frames())
    with pytest.raises(Exception, match="nope"):
        normalize_dataset(reader, ColumnMapping(address_col="nope"), MemorySink(), normalizer=stub())
    assert state["closed"] == 1


def test_reader_closed_on_schema_drift():
    frames = _frames(1) + [pd.DataFrame({"direccion": [ADDR_OK], "extra": ["x"]})]
    reader, state = _chunks(frames)
    with pytest.raises(SourceReadError):
        _normalize(reader)
    assert state["closed"] == 1


def test_reader_closed_when_progress_callback_is_interrupted():
    reader, state = _chunks(_frames())

    def progress(done, chunk_no):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _normalize(reader, progress=progress)
    assert state["closed"] == 1


def test_reader_closed_when_the_source_iterator_itself_fails():
    def gen():
        yield pd.DataFrame({"direccion": [ADDR_OK]})
        raise SourceReadError("boom")

    state = {"closed": 0}
    reader = TableChunks("chunks", ["direccion"], gen(), closers=[lambda: state.update(closed=state["closed"] + 1)])
    with pytest.raises(SourceReadError):
        _normalize(reader)
    assert state["closed"] == 1


@pytest.mark.skipif(os.name != "nt", reason="an open handle only blocks deletion on Windows")
def test_input_file_can_be_deleted_right_after_a_failed_run(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES, ADDR_OK]}))
    with pytest.raises(RuntimeError):
        normalize_dataset(src, ColumnMapping(address_col="direccion"), MemorySink(),
                          normalizer=exploding_stub(), chunk_size=1, on_error="raise")
    os.remove(src)
    assert not os.path.exists(src)


def test_path_source_is_closed_on_mapping_error(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    with pytest.raises(Exception, match="nope"):
        normalize_dataset(src, ColumnMapping(address_col="nope"), MemorySink(), normalizer=stub())
    os.remove(src)  # would raise PermissionError on Windows if the handle leaked


# ---------------------------------------------------------------------------
# pipeline: which exceptions are row errors and which abort the run
# ---------------------------------------------------------------------------
def test_memory_error_aborts_the_run_and_closes_the_reader():
    reader, state = _chunks(_poisoned())
    sink = RecordingSink()
    with pytest.raises(MemoryError):
        _normalize(reader, normalizer=RaisingStub(MemoryError("oom")), sink=sink)
    assert state["closed"] == 1


def test_memory_error_aborts_even_for_a_single_row_batch():
    reader, state = _chunks([pd.DataFrame({"direccion": [ADDR_RAISES]})])
    with pytest.raises(MemoryError):
        _normalize(reader, normalizer=RaisingStub(MemoryError()))
    assert state["closed"] == 1


def test_recursion_error_is_still_a_row_error():
    reader, _ = _chunks(_poisoned())
    sink = MemorySink()
    summary = _normalize(reader, normalizer=RaisingStub(RecursionError("deep")), sink=sink)
    assert summary["error"] == 1 and summary["ok"] == 3
    assert sink.frame.loc[2, "motivo"].startswith("RecursionError")


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3)])
def test_base_exceptions_from_the_normalizer_are_never_swallowed(exc):
    reader, state = _chunks(_poisoned())
    with pytest.raises(type(exc)):
        _normalize(reader, normalizer=RaisingStub(exc))
    assert state["closed"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv, *, normalizer=None, gazetteer_loader=None):
    def factory(artifacts_dir, device):
        return normalizer or stub()

    return cli_module.main(argv, normalizer_factory=factory, gazetteer_loader=gazetteer_loader or (lambda *a, **k: None))


@pytest.fixture
def recorded_reader(monkeypatch):
    """Wrap cli.read_table so tests can see whether the reader it built was closed."""
    made = []
    real = cli_module.read_table

    def wrapper(*args, **kwargs):
        reader = real(*args, **kwargs)
        state = {"closed": 0}
        original = reader.close

        def close():
            state["closed"] += 1
            original()

        reader.close = close
        made.append(state)
        return reader

    monkeypatch.setattr(cli_module, "read_table", wrapper)
    return made


def _partials(tmp_path):
    return [f for f in os.listdir(tmp_path) if f.endswith(".partial")]


@pytest.fixture
def isolated_cli(monkeypatch):
    """The CLI wired to a recording reader and sink, with a pipeline that releases NOTHING itself.

    Only the CLI's own success/``finally`` code can close the reader or abort/close the sink here.
    """
    state = {"reader_closed": 0, "sink": RecordingSink(), "behaviour": None}

    def fake_read_table(source, **kwargs):
        reader = TableChunks("chunks", ["direccion"], iter(()),
                             closers=[lambda: state.__setitem__("reader_closed", state["reader_closed"] + 1)])
        state["reader"] = reader  # keep it alive: TableChunks.__del__ would otherwise close it and mask a missing close
        return reader

    def fake_normalize_dataset(reader, mapping, sink, **kwargs):
        if state["behaviour"] is not None:
            raise state["behaviour"]
        return {"rows": 0, "error": 0, "by_estado": {}, "seconds": 0.0, "rows_per_sec": 0.0}

    monkeypatch.setattr(cli_module, "read_table", fake_read_table)
    monkeypatch.setattr(cli_module, "open_sink", lambda *a, **k: state["sink"])
    monkeypatch.setattr("cali_address.io.pipeline.normalize_dataset", fake_normalize_dataset)
    return state


def test_cli_closes_reader_on_success(tmp_path, isolated_cli):
    assert _cli(["normalize", "ignored.csv", "-o", str(tmp_path / "o.csv")]) == 0
    assert isolated_cli["reader_closed"] >= 1
    assert isolated_cli["sink"].closes == 1 and isolated_cli["sink"].aborts == 0


def test_cli_closes_reader_and_leaves_no_output_on_on_error_raise(tmp_path, recorded_reader, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES]}))
    out = tmp_path / "o.csv"
    assert _cli(["normalize", src, "-o", str(out), "--on-error", "raise", "--chunk-size", "1"],
                normalizer=exploding_stub()) == 1
    assert recorded_reader[0]["closed"] >= 1 and not out.exists() and _partials(tmp_path) == []
    os.remove(src)  # the input is free again


def test_cli_closes_reader_when_the_gazetteer_loader_fails(tmp_path, recorded_reader):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    out = tmp_path / "o.csv"

    def bad_loader(*a, **k):
        raise ValueError("corrupt gazetteer")

    with pytest.raises(ValueError):
        _cli(["normalize", src, "-o", str(out)], gazetteer_loader=bad_loader)
    assert recorded_reader[0]["closed"] >= 1 and not out.exists() and _partials(tmp_path) == []


def test_cli_closes_reader_when_the_sink_fails(tmp_path, recorded_reader, monkeypatch, capsys):
    from cali_address.io.errors import SinkWriteError

    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    out = tmp_path / "o.csv"
    real_open = cli_module.open_sink

    def failing_open(path, fmt=None, **kw):
        sink = real_open(path, fmt, **kw)

        def boom(df):
            raise SinkWriteError("disk full")

        sink.write = boom
        return sink

    monkeypatch.setattr(cli_module, "open_sink", failing_open)
    assert _cli(["normalize", src, "-o", str(out)]) == 3
    assert recorded_reader[0]["closed"] >= 1 and not out.exists() and _partials(tmp_path) == []
    assert "disk full" in capsys.readouterr().err


def test_cli_memory_error_reports_cleanly_without_partial_output(tmp_path, recorded_reader, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES]}))
    out = tmp_path / "o.csv"
    code = _cli(["normalize", src, "-o", str(out), "--chunk-size", "1"], normalizer=RaisingStub(MemoryError()))
    err = capsys.readouterr().err
    assert code == 1 and "memory" in err.lower() and "Traceback" not in err
    assert recorded_reader[0]["closed"] >= 1 and not out.exists() and _partials(tmp_path) == []


def test_cli_keyboard_interrupt_cleans_up_and_exits_130(tmp_path, isolated_cli):
    isolated_cli["behaviour"] = KeyboardInterrupt()
    code = _cli(["normalize", "ignored.csv", "-o", str(tmp_path / "o.csv")])
    assert code == 130
    # the pipeline stub released nothing: the CLI's own ``finally`` must abort the sink and close the reader
    assert isolated_cli["reader_closed"] >= 1
    assert isolated_cli["sink"].aborts == 1 and isolated_cli["sink"].closes == 0


def test_cli_keyboard_interrupt_still_leaves_no_partial_output_with_the_real_pipeline(tmp_path, recorded_reader):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_RAISES]}))
    out = tmp_path / "o.csv"
    code = _cli(["normalize", src, "-o", str(out), "--chunk-size", "1"], normalizer=RaisingStub(KeyboardInterrupt()))
    assert code == 130
    assert recorded_reader[0]["closed"] >= 1 and not out.exists() and _partials(tmp_path) == []


def test_cli_closes_reader_when_the_model_artifacts_are_missing(tmp_path, recorded_reader, capsys):
    from cali_address.paths import ArtifactsMissingError

    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))

    def factory(artifacts_dir, device):
        raise ArtifactsMissingError("missing model.pt")

    code = cli_module.main(["normalize", src, "-o", str(tmp_path / "o.csv")], normalizer_factory=factory,
                           gazetteer_loader=lambda *a, **k: None)
    assert code == 3 and recorded_reader[0]["closed"] >= 1 and _partials(tmp_path) == []


def test_read_table_reader_is_a_context_manager_that_closes(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    with read_table(src):
        pass
    os.remove(src)
