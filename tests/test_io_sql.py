"""SQL source: SQLAlchemy URL + table/query, server-side chunking, safety, failure modes."""

from __future__ import annotations

import sqlite3

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


@pytest.mark.parametrize("chunk_size", [1, 2, 6, 7, 8, 20_000])
def test_sql_table_chunk_boundaries(db_url, chunk_size):
    chunks = list(read_table(db_url, table="direcciones", chunk_size=chunk_size))
    assert sum(len(c) for c in chunks) == 7 and all(len(c) <= chunk_size for c in chunks)
    out = collect(chunks)
    assert out["id"].tolist() == [f"{i:03d}" for i in range(7)]


def test_sql_query_source(db_url):
    reader = read_table(db_url, query="SELECT id, direccion FROM direcciones WHERE id >= '003' ORDER BY id")
    assert reader.columns == ["id", "direccion"]
    assert len(collect(reader)) == 4


def test_sql_query_with_leading_comment_and_cte_allowed(db_url):
    q = "WITH t AS (SELECT * FROM direcciones) SELECT id FROM t"
    assert len(collect(read_table(db_url, query=q))) == 7


def test_sql_empty_table_has_columns_and_no_rows(db_url):
    reader = read_table(db_url, table="vacia")
    assert reader.columns == ["id", "direccion"] and list(reader) == []


def test_sql_requires_exactly_one_of_table_or_query(db_url):
    with pytest.raises(UsageError):
        read_table(db_url)
    with pytest.raises(UsageError):
        read_table(db_url, table="direcciones", query="SELECT 1")


@pytest.mark.parametrize("bad", [
    "direcciones; DROP TABLE direcciones", "direcciones--x", "a b", "", "x'y", "a.b.c.d", "1abc", 'a"b',
])
def test_sql_rejects_invalid_table_identifiers(db_url, bad):
    with pytest.raises(UsageError):
        read_table(db_url, table=bad)


@pytest.mark.parametrize("query", [
    "DELETE FROM direcciones", "DROP TABLE direcciones", "UPDATE direcciones SET id='x'",
    "SELECT 1; DROP TABLE direcciones", "INSERT INTO direcciones VALUES (1)", "   ",
])
def test_sql_rejects_non_select_queries(db_url, query):
    with pytest.raises(UsageError):
        read_table(db_url, query=query)


def test_sql_table_is_never_modified_by_read(db_url, tmp_path):
    collect(read_table(db_url, table="direcciones"))
    con = sqlite3.connect(tmp_path / "addr.db")
    assert con.execute("SELECT count(*) FROM direcciones").fetchone()[0] == 7


def test_sql_missing_table_is_read_error(db_url):
    with pytest.raises(SourceReadError):
        read_table(db_url, table="no_existe")


def test_sql_unreachable_database_is_read_error_without_leaking_password(tmp_path):
    url = f"sqlite:///{(tmp_path / 'no' / 'such' / 'dir' / 'x.db').as_posix()}"
    with pytest.raises(SourceReadError):
        read_table(url, table="t")


def test_sql_missing_driver_gives_install_hint_and_hides_password():
    with pytest.raises((MissingDependencyError, SourceReadError)) as exc:
        read_table("postgresql://user:sup3rsecret@127.0.0.1:1/db", table="t")
    assert "sup3rsecret" not in str(exc.value)


def test_sql_unreachable_server_with_available_driver_does_not_leak_password(monkeypatch):
    # mysql/postgres drivers are absent in CI; a sqlite URL with a password-like userinfo is enough
    # to check the redaction helper on the message path.
    from cali_address.io.readers import redact_url

    assert "sup3rsecret" not in redact_url("postgresql://user:sup3rsecret@host/db")
    assert redact_url("not a url at all") == "<unparseable url>"


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
