"""SQL source: SQLAlchemy URL + --table (table or view), server-side chunking, safety, failure modes."""

from __future__ import annotations

import sqlite3
import urllib.parse

import pytest

from dataset_helpers import ADDR_OK, collect, synth_frame

from cali_address.io import (
    MissingDependencyError,
    SourceReadError,
    UsageError,
    read_table,
)


@pytest.fixture
def db_url(tmp_path):
    path = tmp_path / "addr.db"
    con = sqlite3.connect(path)
    frame = synth_frame(7)
    frame.to_sql("direcciones", con, index=False)
    con.execute('CREATE TABLE "weird name" (direccion TEXT)')
    con.execute("INSERT INTO \"weird name\" VALUES ('CL 5 # 38 - 20')")
    con.execute("CREATE TABLE vacia (id TEXT, direccion TEXT)")
    con.commit()
    con.close()
    return f"sqlite:///{path.as_posix()}"


def _count(tmp_path, table="direcciones"):
    con = sqlite3.connect(tmp_path / "addr.db")
    try:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


@pytest.mark.parametrize("chunk_size", [1, 2, 6, 7, 8, 20_000])
def test_sql_table_chunk_boundaries(db_url, chunk_size):
    chunks = list(read_table(db_url, table="direcciones", chunk_size=chunk_size))
    assert sum(len(c) for c in chunks) == 7 and all(len(c) <= chunk_size for c in chunks)
    out = collect(chunks)
    assert out["id"].tolist() == [f"{i:03d}" for i in range(7)]


def test_sql_view_is_read_like_a_table(db_url, tmp_path):
    """The documented recipe for joins/filters: create a view, read it with --table."""
    con = sqlite3.connect(tmp_path / "addr.db")
    con.execute("CREATE VIEW ultimos AS SELECT id, direccion FROM direcciones WHERE id >= '003' ORDER BY id")
    con.commit()
    con.close()
    reader = read_table(db_url, table="ultimos")
    assert reader.columns == ["id", "direccion"] and len(collect(reader)) == 4


def test_sql_empty_table_has_columns_and_no_rows(db_url):
    reader = read_table(db_url, table="vacia")
    assert reader.columns == ["id", "direccion"] and list(reader) == []


@pytest.mark.parametrize("bad", [
    "direcciones; DROP TABLE direcciones", "direcciones--x", "a b", "", "x'y", "a.b.c.d", "1abc", 'a"b',
    "direcciones/**/", "direcciones\n", "direcciones ", "d.", ".d", "s..d", "`d`", "[d]", "dir\x00",
    "direcciones UNION SELECT 1", "direcciones;", "(SELECT 1)", "main.direcciones; DELETE FROM direcciones",
    "direcci\u00f3n", "d\u00a0x",
])
def test_sql_rejects_invalid_table_identifiers(db_url, tmp_path, bad):
    with pytest.raises(UsageError):
        read_table(db_url, table=bad)
    assert _count(tmp_path) == 7  # nothing ran


@pytest.mark.parametrize("attack", [
    "direcciones; DELETE FROM direcciones", "direcciones; DROP TABLE direcciones",
    "direcciones WHERE 1=1; UPDATE direcciones SET id='x'", "main.direcciones;--",
])
def test_sql_injection_style_table_names_never_modify_the_table(db_url, tmp_path, attack):
    with pytest.raises(UsageError):
        read_table(db_url, table=attack)
    assert _count(tmp_path) == 7


def test_sql_identifiers_are_quoted_by_the_driver_not_interpolated(db_url):
    """A keyword-like but valid identifier is quoted/escaped by SQLAlchemy and simply does not exist."""
    with pytest.raises(SourceReadError):
        read_table(db_url, table="select")
    assert len(collect(read_table(db_url, table="direcciones"))) == 7


# ---------------------------------------------------------------------------
# read-only enforcement
# ---------------------------------------------------------------------------

def test_sql_normal_read_never_commits_and_leaves_no_lock(db_url, tmp_path):
    collect(read_table(db_url, table="direcciones"))
    con = sqlite3.connect(tmp_path / "addr.db", timeout=0.2)
    con.execute("INSERT INTO vacia VALUES ('1', 'x')")  # 'database is locked' if a transaction stayed open
    con.commit()
    con.close()
    assert _count(tmp_path) == 7


def test_sql_missing_table_is_read_error(db_url):
    with pytest.raises(SourceReadError):
        read_table(db_url, table="no_existe")


# ---------------------------------------------------------------------------
# failure modes and credential hygiene
# ---------------------------------------------------------------------------
class _FakeDialect:
    name = "postgresql"


class _FakeEngine:
    """An engine whose server rejects the login and echoes the password back (worst case)."""

    dialect = _FakeDialect()

    def __init__(self, leaks):
        self.leaks = leaks
        self.disposed = False

    def connect(self):
        import sqlalchemy as sa

        raise sa.exc.OperationalError("SELECT 1", {}, Exception(self.leaks))

    def dispose(self):
        self.disposed = True


def _url_for(password):
    return f"postgresql://user:{urllib.parse.quote(password, safe='')}@db.example.org:5432/cali"


def test_sql_unreachable_database_is_read_error(tmp_path):
    url = f"sqlite:///{(tmp_path / 'no' / 'such' / 'dir' / 'x.db').as_posix()}"
    with pytest.raises(SourceReadError, match="SQLite database file not found"):
        read_table(url, table="t")
    assert not (tmp_path / "no").exists()


@pytest.mark.parametrize("password", ["sup3rsecret", "p@ss", "a/b:c@d", "50%off"])
def test_sql_unreachable_server_never_leaks_raw_or_url_encoded_password(monkeypatch, caplog, capsys, password):
    import sqlalchemy as sa

    encoded = urllib.parse.quote(password, safe="")
    leak = f"FATAL: auth failed (tried {password!r}, {password}, {encoded}, url {_url_for(password)})"
    engine = _FakeEngine(leak)
    monkeypatch.setattr(sa, "create_engine", lambda url, **kw: engine)
    with caplog.at_level("DEBUG"):
        with pytest.raises(SourceReadError) as exc:
            read_table(_url_for(password), table="t")
    captured = capsys.readouterr()
    text = str(exc.value) + caplog.text + captured.out + captured.err
    assert "***" in str(exc.value) and "database query failed" in str(exc.value)
    assert password not in text and encoded not in text
    assert engine.disposed


def test_sql_missing_driver_gives_install_hint_and_hides_password():
    with pytest.raises((MissingDependencyError, SourceReadError)) as exc:
        read_table("postgresql://user:p%40ss@127.0.0.1:1/db", table="t")
    assert "p%40ss" not in str(exc.value) and "p@ss" not in str(exc.value)


def test_redact_url_masks_password_and_survives_garbage():
    from cali_address.io.readers import redact_url

    assert "sup3rsecret" not in redact_url("postgresql://user:sup3rsecret@host/db")
    assert "p%40ss" not in redact_url("postgresql://user:p%40ss@host/db")
    assert redact_url("not a url at all") == "<unparseable url>"


def test_short_masks_every_form_of_the_secret():
    from cali_address.io.readers import _short

    class Boom(Exception):
        pass

    msg = _short(Boom("bad p@ss and p%40ss and P%40SS"), ["p@ss", "p%40ss"])
    assert "p@ss" not in msg and "p%40ss" not in msg and "P%40SS" not in msg
    assert _short(Boom("plain"), None) == "plain"


def test_sql_values_with_quotes_round_trip(tmp_path):
    path = tmp_path / "q.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE d (direccion TEXT)")
    con.execute("INSERT INTO d VALUES (?)", ("Calle 5 # 38-20 O'Brien \"x\"",))
    con.commit()
    con.close()
    out = collect(read_table(f"sqlite:///{path.as_posix()}", table="d"))
    assert out.loc[0, "direccion"] == "Calle 5 # 38-20 O'Brien \"x\""


def test_sql_schema_qualified_identifier_is_accepted(db_url):
    # sqlite exposes the attached "main" schema
    assert len(collect(read_table(db_url, table="main.direcciones"))) == 7


def test_sql_null_values_become_missing(tmp_path):
    path = tmp_path / "n.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE d (direccion TEXT)")
    con.execute("INSERT INTO d VALUES (NULL)")
    con.commit()
    con.close()
    out = collect(read_table(f"sqlite:///{path.as_posix()}", table="d"))
    assert out["direccion"].isna().all()
    assert ADDR_OK  # keep import used
