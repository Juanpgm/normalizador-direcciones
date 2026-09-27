"""SQL sources are table/view based only: read-only sessions, cleanup, credential hygiene, removed query feature.

The free-form ``--sql-query`` feature was removed on purpose: a text filter cannot be a security boundary for SQL
(three review rounds kept finding dialect-specific bypasses). These tests pin what remains and that it stays gone.
"""

from __future__ import annotations

import sqlite3

import pytest

from dataset_helpers import stub

from cali_address.cli import main
from cali_address.io import MissingDependencyError, SourceReadError, UsageError, read_table


def _run(argv):
    made = []

    def factory(artifacts_dir, device):
        made.append(1)
        return stub()

    return main(argv, normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None), made


# ---------------------------------------------------------------------------
# the free-form query feature is gone: flags, config keys and the Python parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flags", [
    ["--sql-query", "SELECT 1"], ["--sql-query=SELECT 1"], ["--trust-query"], ["--no-trust-query"],
    ["--table", "t", "--sql-query", "SELECT 1", "--trust-query"],
], ids=["sql-query", "sql-query-eq", "trust-query", "no-trust-query", "table-plus-both"])
@pytest.mark.parametrize("command", ["normalize", "inspect"])
def test_removed_flags_are_rejected_by_argparse_with_exit_2(command, flags, tmp_path, capsys):
    argv = [command, "sqlite:///x.db", *flags]
    if command == "normalize":
        argv += ["-o", str(tmp_path / "o.csv")]
    with pytest.raises(SystemExit) as exc:
        _run(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err and "Traceback" not in err


@pytest.mark.parametrize("line", ['sql_query = "SELECT 1"', "trust_query = true", 'trust_query = "yes"'])
def test_removed_config_keys_are_unknown_options(line, tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'input = "sqlite:///x.db"\ntable = "t"\n{line}\n', encoding="utf-8")
    code, made = _run(["normalize", "--config", str(cfg), "-o", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert code == 2 and made == []
    assert "unknown config option(s)" in err and line.split("=")[0].strip() in err


def test_read_table_no_longer_accepts_query_parameters(tmp_path):
    path = tmp_path / "a.db"
    sqlite3.connect(path).close()
    url = f"sqlite:///{path.as_posix()}"
    with pytest.raises(TypeError):
        read_table(url, query="SELECT 1")
    with pytest.raises(TypeError):
        read_table(url, table="t", trust_query=True)


def test_a_sql_source_without_a_table_is_a_usage_error(tmp_path):
    path = tmp_path / "a.db"
    sqlite3.connect(path).close()
    with pytest.raises(UsageError, match="--table"):
        read_table(f"sqlite:///{path.as_posix()}")


def test_invalid_table_message_points_to_views_not_to_a_query_flag():
    with pytest.raises(UsageError) as exc:
        read_table("sqlite://", table="a b")
    assert "sql-query" not in str(exc.value) and "view" in str(exc.value).lower()


def test_validator_machinery_is_gone_from_the_reader_module():
    from cali_address.io import readers

    for name in ("_validate_query", "_strip_sql_noise", "_AMBIGUOUS_SYNTAX", "_WRITE_KEYWORDS", "_LOCKING_READ",
                 "_require_read_only_session", "_READ_ONLY_POSTGRES_DRIVERS", "_READ_ONLY_DIALECTS"):
        assert not hasattr(readers, name), name


# ---------------------------------------------------------------------------
# sqlite FILE database: the read-only guard is on every pooled connection
# ---------------------------------------------------------------------------
def _sqlite_file(tmp_path):
    path = tmp_path / "ro.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return path


def test_sqlite_file_guard_is_active_on_every_pooled_connection(tmp_path, monkeypatch):
    """A write cannot happen through a table read: every pooled DBAPI connection has query_only=ON."""
    import sqlalchemy as sa

    path = _sqlite_file(tmp_path)
    seen = []
    real = sa.create_engine

    def spy(*a, **k):
        engine = real(*a, **k)
        seen.append(engine)
        return engine

    monkeypatch.setattr(sa, "create_engine", spy)
    reader = read_table(f"sqlite:///{path.as_posix()}", table="t")
    try:
        engine = seen[0]
        raw = [engine.raw_connection() for _ in range(3)]  # distinct DBAPI connections from the same pool
        try:
            assert len({id(r.dbapi_connection) for r in raw}) == 3
            for conn in raw:
                assert conn.cursor().execute("PRAGMA query_only").fetchone()[0] == 1
                with pytest.raises(sqlite3.OperationalError, match="(?i)readonly|read-only|read only"):
                    conn.cursor().execute("INSERT INTO t VALUES (99)")
        finally:
            for conn in raw:
                conn.close()
    finally:
        reader.close()
    con = sqlite3.connect(path)
    try:
        assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 1
    finally:
        con.close()


# ---------------------------------------------------------------------------
# fakes for the session/cleanup contract (no network, no driver needed)
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, log, close_error=None):
        self.log, self.close_error = log, close_error

    def keys(self):
        return ["id", "direccion"]

    def fetchmany(self, n):
        return []

    def close(self):
        self.log.append("result.close")
        if self.close_error:
            raise self.close_error


class _FakeConnection:
    def __init__(self, log, *, close_error=None, result_close_error=None, rollback_error=None, fail_execute=None):
        self.log = log
        self.close_error, self.result_close_error = close_error, result_close_error
        self.rollback_error, self.fail_execute = rollback_error, fail_execute
        self.options: dict = {}

    def execution_options(self, **options):
        self.log.append(("execution_options", dict(options)))
        self.options = options
        return self

    def execute(self, statement):
        self.log.append("execute")
        if self.fail_execute:
            raise self.fail_execute
        return _FakeResult(self.log, self.result_close_error)

    def rollback(self):
        self.log.append("rollback")
        if self.rollback_error:
            raise self.rollback_error

    def close(self):
        self.log.append("connection.close")
        if self.close_error:
            raise self.close_error


class _FakeEngine:
    def __init__(self, connection, dialect, log):
        self._connection, self.dialect, self.log = connection, dialect, log
        self.disposed = 0

    def connect(self):
        self.log.append("connect")
        return self._connection

    def dispose(self):
        self.log.append("engine.dispose")
        self.disposed += 1


def _install(monkeypatch, connection, dialect, log):
    import sqlalchemy as sa

    engine = _FakeEngine(connection, dialect, log)
    urls = []

    def create(url, **kw):
        urls.append(url)
        return engine

    monkeypatch.setattr(sa, "create_engine", create)
    return engine, urls


_PG = "postgresql+psycopg2://u:pw@h/db"  # fakes stand in for a server; no sqlite listener needed


def _dialect_for(url):
    from sqlalchemy.engine import make_url

    return make_url(url).get_dialect()()


# ---------------------------------------------------------------------------
# cleanup: dispose ALWAYS runs, and the original error wins
# ---------------------------------------------------------------------------
def test_engine_is_disposed_even_when_connection_close_raises(monkeypatch):
    log: list = []
    conn = _FakeConnection(log, close_error=RuntimeError("close boom"))
    engine, _ = _install(monkeypatch, conn, _dialect_for(_PG), log)
    reader = read_table(_PG, table="t")
    with pytest.raises(RuntimeError, match="close boom"):  # the closer itself reports; TableChunks.close swallows it
        reader._closers[0]()
    assert engine.disposed == 1
    assert log.index("rollback") < log.index("connection.close") < log.index("engine.dispose")


def test_reader_close_never_raises_and_still_disposes(monkeypatch):
    log: list = []
    conn = _FakeConnection(log, close_error=RuntimeError("close boom"))
    engine, _ = _install(monkeypatch, conn, _dialect_for(_PG), log)
    reader = read_table(_PG, table="t")
    reader.close()
    reader.close()  # idempotent
    assert engine.disposed == 1


def test_first_cleanup_error_wins_and_every_step_still_runs(monkeypatch):
    log: list = []
    conn = _FakeConnection(log, result_close_error=ValueError("result first"), close_error=RuntimeError("later"))
    engine, _ = _install(monkeypatch, conn, _dialect_for(_PG), log)
    reader = read_table(_PG, table="t")
    with pytest.raises(ValueError, match="result first"):
        reader._closers[0]()
    assert "rollback" in log and "connection.close" in log and engine.disposed == 1


def test_rollback_failure_does_not_skip_close_or_dispose(monkeypatch):
    log: list = []
    conn = _FakeConnection(log, rollback_error=RuntimeError("rollback boom"))
    engine, _ = _install(monkeypatch, conn, _dialect_for(_PG), log)
    read_table(_PG, table="t").close()
    assert "connection.close" in log and engine.disposed == 1


def test_failed_query_reports_the_query_error_even_if_cleanup_also_fails(monkeypatch):
    import sqlalchemy as sa

    log: list = []
    conn = _FakeConnection(
        log, close_error=RuntimeError("close boom"),
        fail_execute=sa.exc.OperationalError("SELECT 1", {}, Exception("no such table: t")),
    )
    engine, _ = _install(monkeypatch, conn, _dialect_for(_PG), log)
    with pytest.raises(SourceReadError, match="database query failed") as exc:
        read_table(_PG, table="t")
    assert "no such table" in str(exc.value) and "close boom" not in str(exc.value)
    assert engine.disposed == 1


def test_connect_failure_disposes_the_engine(monkeypatch):
    import sqlalchemy as sa

    log: list = []
    engine, _ = _install(monkeypatch, None, _dialect_for(_PG), log)
    engine.connect = lambda: (_ for _ in ()).throw(sa.exc.OperationalError("x", {}, Exception("refused")))
    with pytest.raises(SourceReadError):
        read_table(_PG, table="t")
    assert engine.disposed == 1


# ---------------------------------------------------------------------------
# PostgreSQL: postgresql_readonly=True is applied BEFORE any statement, where the driver implements it.
# Decided from the installed SQLAlchemy sources: psycopg2, psycopg, pg8000 (and asyncpg / psycopg_async, whose
# dialects also define set_readonly) implement it; the base PGDialect raises NotImplementedError, so a dialect that
# does not override set_readonly (third-party drivers) simply does not get the option and nothing fails.
# ---------------------------------------------------------------------------
PG_IMPLEMENTING = [
    "postgresql://u:pw@h/db", "postgresql+psycopg2://u:pw@h/db", "postgresql+pg8000://u:pw@h/db",
    "postgresql+psycopg://u:pw@h/db", "postgresql+asyncpg://u:pw@h/db", "postgresql+psycopg_async://u:pw@h/db",
]


@pytest.mark.parametrize("url", PG_IMPLEMENTING)
def test_postgresql_readonly_is_applied_before_any_statement(url, monkeypatch):
    log: list = []
    conn = _FakeConnection(log)
    _, urls = _install(monkeypatch, conn, _dialect_for(url), log)
    read_table(url, table="public.direcciones").close()
    assert urls == [url]
    options = [e for e in log if isinstance(e, tuple)]
    assert len(options) == 1 and options[0][1].get("postgresql_readonly") is True
    assert log.index(options[0]) < log.index("execute")  # the session is read-only before the SELECT runs
    assert log.index("connect") < log.index(options[0])


def test_a_postgresql_dialect_without_set_readonly_support_gets_no_readonly_option_and_no_crash(monkeypatch):
    from sqlalchemy.dialects.postgresql.base import PGDialect

    class _NoReadOnlyDialect(PGDialect):  # what a third-party driver dialect looks like: base set_readonly only
        pass

    log: list = []
    conn = _FakeConnection(log)
    _install(monkeypatch, conn, _NoReadOnlyDialect(), log)
    reader = read_table("postgresql+thirdparty://u:pw@h/db", table="t")
    options = [e for e in log if isinstance(e, tuple)][0][1]
    assert "postgresql_readonly" not in options and options.get("stream_results") is True
    reader.close()


@pytest.mark.parametrize("url", ["mysql+pymysql://u:pw@h/db", "mssql+pyodbc://u:pw@h/db", "oracle://u:pw@h/db"])
def test_other_dialects_never_get_the_postgresql_option(url, monkeypatch):
    class _Dialect:
        name = url.split(":")[0].split("+")[0]

    log: list = []
    _install(monkeypatch, _FakeConnection(log), _Dialect(), log)
    read_table(url, table="t").close()
    assert "postgresql_readonly" not in [e for e in log if isinstance(e, tuple)][0][1]


def test_a_read_never_commits(monkeypatch):
    log: list = []
    conn = _FakeConnection(log)

    def commit():
        log.append("COMMIT")
        pytest.fail("a read must never commit")  # Failed is not an Exception: the reader's cleanup cannot swallow it

    conn.commit = commit
    _install(monkeypatch, conn, _dialect_for(_PG), log)
    reader = read_table(_PG, table="t")
    reader.close()
    assert "COMMIT" not in log
    assert log.count("rollback") == 1 and log.index("rollback") < log.index("connection.close")


# ---------------------------------------------------------------------------
# Every URL / dialect is allowed for table reads (no gating any more)
# ---------------------------------------------------------------------------
TABLE_READ_URLS = [
    "mysql+pymysql://u:pw@127.0.0.1:1/db", "mariadb+pymysql://u:pw@127.0.0.1:1/db",
    "mssql+pyodbc://u:pw@127.0.0.1:1/db", "oracle://u:pw@127.0.0.1:1/db", "postgresql://u:pw@127.0.0.1:1/db",
]


@pytest.mark.parametrize("url", TABLE_READ_URLS)
def test_table_reads_are_allowed_for_every_dialect_missing_driver(url, monkeypatch):
    """No dialect is rejected up front: a missing driver is the ONLY reason to refuse, with an install hint."""
    import sqlalchemy as sa

    def create(u, **kw):
        raise ImportError("No module named 'driver'")

    monkeypatch.setattr(sa, "create_engine", create)
    with pytest.raises(MissingDependencyError, match="is not installed") as exc:
        read_table(url, table="t")
    assert "install" in str(exc.value) and ":pw@" not in str(exc.value) and "trust" not in str(exc.value).lower()


@pytest.mark.parametrize("url", TABLE_READ_URLS)
def test_table_reads_are_allowed_for_every_dialect_unreachable_server(url, monkeypatch):
    """With the driver present, the only failure is the server itself: a SourceReadError (exit 3), password masked."""
    import sqlalchemy as sa

    class _Unreachable(_FakeEngine):
        def connect(self):
            raise sa.exc.OperationalError("connect", {}, Exception(f"could not connect to {url}"))

    log: list = []
    dialect = type("_Dialect", (), {"name": url.split(":")[0].split("+")[0]})()  # driver-less stand-in
    engine = _Unreachable(_FakeConnection(log), dialect, log)
    monkeypatch.setattr(sa, "create_engine", lambda u, **kw: engine)
    with pytest.raises(SourceReadError, match="database query failed") as exc:
        read_table(url, table="t")
    assert ":pw@" not in str(exc.value) and "trust" not in str(exc.value).lower()
    assert engine.disposed == 1


# ---------------------------------------------------------------------------
# An unparseable URL is a usage error (exit 2), redacted, with no traceback
# ---------------------------------------------------------------------------
GARBAGE_URLS = {
    "words": "not a url",
    "scheme-separator-only": "://",
    "empty-authority": "postgresql://:@:/",
    "empty-string": "",
    "scheme-only": "postgresql:",
    "unknown-dialect": "foo://x",
    "unknown-dialect-with-secret": "foo://user:s3cretpw@host/db",
    "bad-port-with-secret": "postgresql://user:s3cretpw@host:notaport/db",
    "postgres-alias": "postgres://user:s3cretpw@host/db",
    "unbalanced-bracket-with-secret": "weird+://bob:s3cretpw@[bad/x",
}


@pytest.mark.parametrize("url", list(GARBAGE_URLS.values()), ids=list(GARBAGE_URLS))
def test_unparseable_url_is_a_usage_error_without_password(url):
    with pytest.raises(UsageError) as exc:
        read_table(url, fmt="sql", table="t")
    assert "s3cretpw" not in str(exc.value) and "Traceback" not in str(exc.value)
    assert "s3cretpw" not in repr(exc.value.__cause__) and "s3cretpw" not in repr(exc.value.__context__)


@pytest.mark.parametrize("url", list(GARBAGE_URLS.values()), ids=list(GARBAGE_URLS))
def test_cli_garbage_url_exits_2_without_traceback_or_password(url, tmp_path, capsys):
    code, made = _run(["normalize", url, "--format", "sql", "--table", "t", "-o", str(tmp_path / "o.csv")])
    out = capsys.readouterr()
    assert code == 2 and made == []
    assert "s3cretpw" not in out.out + out.err and "Traceback" not in out.out + out.err


# ---------------------------------------------------------------------------
# Short passwords: the documented residual, pinned both ways
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("secret", ["p", "ab"])
def test_short_password_residual_is_pinned_both_ways(secret):
    from cali_address.io.readers import _short, redact_url

    msg = _short(Exception(f"auth failed, password is {secret} (url postgresql://u:{secret}@h/db)"), [secret])
    assert f":{secret}@" not in msg and ":***@" in msg  # masked structurally, inside a URL's userinfo
    assert f"password is {secret} " in msg  # NOT masked in free text: masking 1-2 chars would shred the message
    assert f":{secret}@" not in redact_url(f"postgresql://u:{secret}@h/db")


def _docs_text():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    return " ".join((root / "docs" / "any-dataset.md").read_text(encoding="utf-8").split())


def test_docs_state_the_short_password_residual_and_the_exit_codes():
    flat = _docs_text()
    assert "1-2 character" in flat and "at least 3 characters" in flat
    for code in ("`0`", "`1`", "`2`", "`3`", "`4`", "`130`"):
        assert code in flat, code


def test_docs_describe_the_table_and_view_recipe_and_the_read_only_session():
    flat = _docs_text()
    assert "--table" in flat and "VIEW" in flat
    assert "psycopg2" in flat and "pg8000" in flat and "read-only database account" in flat
    assert "setval" not in flat


# ---------------------------------------------------------------------------
# A read never creates a SQLite file
# ---------------------------------------------------------------------------
def _files(directory):
    return sorted(p.name for p in directory.iterdir())


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (direccion TEXT)")
    con.execute("INSERT INTO t VALUES ('CL 5 # 38 - 20')")
    con.commit()
    con.close()


def test_reading_a_missing_sqlite_file_fails_and_creates_nothing(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(SourceReadError, match="SQLite database file not found") as exc:
        read_table(f"sqlite:///{path.as_posix()}", table="t")
    assert "missing.db" in str(exc.value)
    assert _files(tmp_path) == []


def test_relative_sqlite_path_is_resolved_against_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SourceReadError, match="SQLite database file not found"):
        read_table("sqlite:///rel.db", table="t")
    assert _files(tmp_path) == []
    _make_db(tmp_path / "rel.db")
    assert len(next(iter(read_table("sqlite:///rel.db", table="t")))) == 1
    assert _files(tmp_path) == ["rel.db"]


def test_missing_sqlite_file_with_query_params_and_a_windows_drive_path_fails_cleanly(tmp_path):
    path = tmp_path / "gone.db"
    with pytest.raises(SourceReadError, match="not found"):
        read_table(f"sqlite:///{path.as_posix()}?mode=ro", table="t")
    with pytest.raises(SourceReadError, match="not found"):
        read_table("sqlite:///Z:/no/such/dir/paths.db", table="t")
    assert _files(tmp_path) == []


def test_existing_sqlite_file_with_a_query_param_still_works(tmp_path):
    path = tmp_path / "ok.db"
    _make_db(path)
    reader = read_table(f"sqlite:///{path.as_posix()}?mode=ro", table="t")
    assert len(next(iter(reader))) == 1
    reader.close()
    assert _files(tmp_path) == ["ok.db"]


def test_missing_table_in_an_existing_sqlite_file_is_a_clean_error_and_adds_no_files(tmp_path):
    path = tmp_path / "ok.db"
    _make_db(path)
    before = _files(tmp_path)
    with pytest.raises(SourceReadError, match="database query failed"):
        read_table(f"sqlite:///{path.as_posix()}", table="no_such_table")
    assert _files(tmp_path) == before


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///:memory:"])
def test_in_memory_sqlite_urls_keep_working(url, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SourceReadError, match="database query failed"):  # empty memory db: no table, but no "file not found"
        read_table(url, table="t")
    assert _files(tmp_path) == []


def test_cli_exits_3_for_a_missing_sqlite_file_and_leaves_it_absent(tmp_path, capsys):
    path = tmp_path / "missing.db"
    for argv in (["inspect", f"sqlite:///{path.as_posix()}", "--table", "t"],
                 ["normalize", f"sqlite:///{path.as_posix()}", "--table", "t", "-o", str(tmp_path / "o.csv")]):
        code, made = _run(argv)
        assert code == 3 and made == []
        assert "SQLite database file not found" in capsys.readouterr().err
    assert _files(tmp_path) == []


# ---------------------------------------------------------------------------
# Identifiers: exactly letters, digits and underscores
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["a$b", "t$", "s$.t", "s.t$x"])
def test_dollar_is_not_allowed_in_identifiers(name):
    with pytest.raises(UsageError, match="letters, digits and underscores only"):
        read_table("sqlite://", table=name)
