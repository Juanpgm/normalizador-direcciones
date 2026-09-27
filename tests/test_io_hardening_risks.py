"""Review follow-ups: cleanup never masks the original error, URL redaction everywhere, ~ in config, short secrets."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from dataset_helpers import ADDR_OK, stub, write_csv

import cali_address.cli as cli
from cali_address.cli import main
from cali_address.io import read_table
from cali_address.io.config import resolve_config_paths
from cali_address.io.readers import _short, redact_url


# ---------------------------------------------------------------------------
# R1: sink.abort() raising must not mask the error nor skip reader.close()
# ---------------------------------------------------------------------------
class _BadSink:
    def __init__(self):
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1
        raise OSError("disk full while flushing")

    def close(self):  # pragma: no cover - never reached in these tests
        pass


def _patched(monkeypatch, tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    state = {"closes": 0, "sink": _BadSink()}
    real = cli.read_table

    def fake_read_table(*a, **k):
        reader = real(*a, **k)
        original = reader.close

        def close():
            state["closes"] += 1
            original()

        reader.close = close
        return reader

    monkeypatch.setattr(cli, "read_table", fake_read_table)
    monkeypatch.setattr(cli, "open_sink", lambda *a, **k: state["sink"])
    return src, state


def test_abort_failure_neither_masks_the_original_error_nor_skips_reader_close(monkeypatch, tmp_path):
    src, state = _patched(monkeypatch, tmp_path)

    def factory(artifacts_dir, device):
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        main(["normalize", str(src), "-o", str(tmp_path / "o.csv")],
             normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None)
    assert state["sink"].abort_calls == 1 and state["closes"] >= 1


def test_abort_failure_still_returns_the_normal_exit_code_and_closes_the_reader(monkeypatch, tmp_path, capsys):
    src, state = _patched(monkeypatch, tmp_path)

    def factory(artifacts_dir, device):
        raise FileNotFoundError("no artifacts here")

    code = main(["normalize", str(src), "-o", str(tmp_path / "o.csv")],
                normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None)
    assert code == 3 and "no artifacts here" in capsys.readouterr().err
    assert state["sink"].abort_calls == 1 and state["closes"] >= 1


# ---------------------------------------------------------------------------
# R2: a password in the source URL is never printed or serialised
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("password", ["sup3rsecret", "p%40ss%2Fx"])
@pytest.mark.parametrize("as_json", [False, True])
def test_inspect_never_prints_the_url_password(monkeypatch, tmp_path, capsys, password, as_json):
    csv = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    real = read_table
    monkeypatch.setattr(cli, "read_table", lambda source, **k: real(str(csv), **{**k, "fmt": "csv"}))
    url = f"postgresql://cali:{password}@db.example.org:5432/cali"
    code = main(["inspect", url, "--table", "t"] + (["--json"] if as_json else []))
    out = capsys.readouterr()
    assert code == 0
    text = out.out + out.err
    assert password not in text and "db.example.org" in text
    if as_json:
        assert json.loads(out.out)["source"].startswith("postgresql://cali:")


def test_inspect_leaves_plain_paths_untouched(tmp_path, capsys):
    csv = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    assert main(["inspect", str(csv), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == str(csv)


def test_redact_source_masks_http_userinfo_and_falls_back_on_garbage():
    from cali_address.io.readers import redact_source

    assert "s3cret" not in redact_source("https://bob:s3cret@example.org/data.csv")
    assert redact_source("/tmp/x.csv") == "/tmp/x.csv"
    assert "s3cret" not in redact_source("weird+://bob:s3cret@[bad/x")


def test_download_error_does_not_leak_the_url_password():
    from cali_address.io import SourceReadError

    with pytest.raises(SourceReadError) as exc:
        read_table("http://bob:s3cretpw@127.0.0.1:1/data.csv")
    assert "s3cretpw" not in str(exc.value)


# ---------------------------------------------------------------------------
# R3: ~ in config path options
# ---------------------------------------------------------------------------
def test_config_tilde_is_expanded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    base = tmp_path / "cfg"
    out = resolve_config_paths({"artifacts_dir": "~/models", "basemaps": "~", "input": "rel.csv"}, str(base))
    assert out["artifacts_dir"] == os.path.normpath(str(home / "models"))
    assert out["basemaps"] == os.path.normpath(str(home))
    assert out["input"] == os.path.normpath(str(base / "rel.csv"))


def test_config_tilde_only_expands_a_leading_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    out = resolve_config_paths({"input": "data/~x.csv", "output": "a~/b.csv"}, str(tmp_path))
    assert out["input"] == os.path.normpath(str(tmp_path / "data" / "~x.csv"))
    assert out["output"] == os.path.normpath(str(tmp_path / "a~" / "b.csv"))


# ---------------------------------------------------------------------------
# R5: short secrets
# ---------------------------------------------------------------------------
class Boom(Exception):
    pass


@pytest.mark.parametrize("secret", ["p", "ab"])
def test_short_secret_keeps_free_text_readable_but_is_masked_inside_urls(secret):
    msg = _short(Boom(f"Password authentication failed for user; url postgresql://cali:{secret}@db/x"), [secret])
    assert msg.startswith("Password authentication failed for user;")
    assert f":{secret}@" not in msg and ":***@" in msg


def test_three_char_secret_is_still_masked_in_free_text():
    assert "abc" not in _short(Boom("bad abc and ABC"), ["abc"])


def test_redact_url_masks_a_one_char_password():
    assert ":p@" not in redact_url("postgresql://u:p@h/db")


# ===========================================================================
# Credential leak regression table: passwords x surfaces
# ===========================================================================
import traceback  # noqa: E402

# id -> (password exactly as written in the URL, strings that must never appear, structural-only?)
# "structural-only" = 1-2 chars: masked inside a URL's userinfo, deliberately NOT in free text (documented residual).
PASSWORDS = {
    "plain": ("sup3rsecret", ["sup3rsecret"], False),
    "pct-at": ("p%40ss", ["p%40ss", "p@ss"], False),
    "pct-slash": ("p%2Fss", ["p%2Fss", "p/ss"], False),
    "pct-colon-hash-qmark-amp": ("a%3Ab%23c%3Fd%26e", ["a%3Ab%23c%3Fd%26e", "a:b#c?d&e"], False),
    "raw-at": ("p@ssw0rd!", ["p@ssw0rd!"], False),
    "raw-colon": ("pa:ssw0rd", ["pa:ssw0rd"], False),
    "raw-slash": ("pa/ssw0rd", ["pa/ssw0rd"], False),
    "raw-hash": ("pa#ssw0rd", ["pa#ssw0rd"], False),
    "raw-qmark": ("pa?ssw0rd", ["pa?ssw0rd"], False),
    "raw-amp": ("pa&ssw0rd", ["pa&ssw0rd"], False),
    "raw-all-specials": ("p@:/#?&w", ["p@:/#?&w"], False),
    "unicode": ("contraseña☃", ["contraseña☃"], False),
    "unicode-encoded": ("contrase%C3%B1a", ["contrase%C3%B1a", "contraseña"], False),
    "three-chars": ("xyz", ["xyz"], False),
    "two-chars": ("zq", [":zq@"], True),
    "one-char": ("q", [":q@"], True),
}
PW_IDS = list(PASSWORDS)


def _url(pw: str, host: str = "db.example.org") -> str:
    return f"postgresql://cali:{pw}@{host}:5432/cali"


def _needles(pw_id: str) -> list[str]:
    return PASSWORDS[pw_id][1]


def _assert_no_leak(text: str, pw_id: str) -> None:
    for needle in _needles(pw_id):
        assert needle not in text, f"{needle!r} leaked in: {text}"


@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_redact_source_and_redact_url(pw_id):
    from cali_address.io.readers import redact_source

    url = _url(PASSWORDS[pw_id][0])
    _assert_no_leak(redact_source(url), pw_id)
    _assert_no_leak(redact_url(url), pw_id)
    assert "db.example.org" in redact_source(url)


@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_short_masks_urls_and_free_text(pw_id):
    from cali_address.io.readers import _url_secrets

    pw, needles, structural_only = PASSWORDS[pw_id]
    url = _url(pw)
    msg = _short(Boom(f"connection to {url} failed"), _url_secrets(url))
    _assert_no_leak(msg, pw_id)
    if not structural_only:  # a copy of the secret outside any URL is masked too
        free = _short(Boom(f"FATAL: password authentication failed, password={pw}"), _url_secrets(url))
        _assert_no_leak(free, pw_id)


def _fake_engine_echoing(url: str, pw: str, monkeypatch):
    """A create_engine whose connect fails with a driver-style message that echoes the URL and the password."""
    import sqlalchemy as sa
    from sqlalchemy import exc as sa_exc
    from urllib.parse import quote

    class Engine:
        dialect = type("D", (), {"name": "postgresql"})()

        def connect(self):
            raise sa_exc.OperationalError(
                f"SELECT 1 -- {url}", {"password": pw},
                Exception(f"could not connect to {url}: FATAL password authentication failed ({pw} / {quote(pw, safe='')})"),
            )

        def dispose(self):
            pass

    monkeypatch.setattr(sa, "create_engine", lambda *a, **k: Engine())


def _chain_text(exc: BaseException) -> str:
    """Everything an exception exposes: the rendered traceback plus str() of every __cause__/__context__ link."""
    parts = ["".join(traceback.format_exception(type(exc), exc, exc.__traceback__))]
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        parts += [str(e), repr(e), *map(str, getattr(e, "args", ()))]
        stack += [e.__cause__, e.__context__]
    return "\n".join(parts)


@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_exception_chain_when_the_driver_echoes_the_password(pw_id, monkeypatch):
    from cali_address.io import SourceReadError

    pw = PASSWORDS[pw_id][0]
    url = _url(pw)
    _fake_engine_echoing(url, pw, monkeypatch)
    with pytest.raises(SourceReadError) as exc:
        read_table(url, table="t")
    text = _chain_text(exc.value)
    if PASSWORDS[pw_id][2]:  # 1-2 chars: only the structural form (inside a URL) is guaranteed masked
        assert f":{pw}@" not in text
    else:
        _assert_no_leak(text, pw_id)


@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_cli_stderr_and_stdout_on_an_unreachable_database(pw_id, tmp_path, capsys):
    pw = PASSWORDS[pw_id][0]
    url = f"postgresql://cali:{pw}@127.0.0.1:1/cali"
    code = main(["normalize", url, "--table", "t", "-o", str(tmp_path / "o.csv")],
                normalizer_factory=lambda *a: stub(), gazetteer_loader=lambda *a, **k: None)
    out = capsys.readouterr()
    assert code in (2, 3)
    text = out.out + out.err
    assert "Traceback" not in text
    if PASSWORDS[pw_id][2]:
        assert f":{pw}@" not in text
    else:
        _assert_no_leak(text, pw_id)


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_inspect_text_and_json(pw_id, as_json, monkeypatch, tmp_path, capsys):
    csv = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    real = read_table
    monkeypatch.setattr(cli, "read_table", lambda source, **k: real(str(csv), **{**k, "fmt": "csv"}))
    code = main(["inspect", _url(PASSWORDS[pw_id][0]), "--table", "t"] + (["--json"] if as_json else []))
    out = capsys.readouterr()
    assert code == 0
    _assert_no_leak(out.out + out.err, pw_id)
    if as_json:
        assert json.loads(out.out)["source"].startswith("postgresql://cali:")


@pytest.mark.parametrize("pw_id", PW_IDS)
def test_leak_summary_json_and_run_output(pw_id, monkeypatch, tmp_path, capsys):
    csv = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    real = read_table
    monkeypatch.setattr(cli, "read_table", lambda source, **k: real(str(csv), **{**k, "fmt": "csv"}))
    sj = tmp_path / "s.json"
    code = main(["normalize", _url(PASSWORDS[pw_id][0]), "--table", "t", "-o", str(tmp_path / "o.csv"),
                 "--summary-json", str(sj)], normalizer_factory=lambda *a: stub(), gazetteer_loader=lambda *a, **k: None)
    out = capsys.readouterr()
    assert code == 0
    _assert_no_leak(out.out + out.err + sj.read_text(encoding="utf-8"), pw_id)


# --- query-string secrets (not covered by SQLAlchemy's hide_password) --------------------------------------
QUERY_URLS = {
    "password": ("postgresql://db.example.org/cali?password=Qs3cret", ["Qs3cret"]),
    "sslpassword": ("postgresql://cali@db.example.org/cali?sslpassword=Qs3cret&sslmode=require", ["Qs3cret"]),
    "user-and-password": ("postgresql://db.example.org/cali?user=cali&password=Qs3cret", ["Qs3cret"]),
    "uppercase-key": ("postgresql://db.example.org/cali?PASSWORD=Qs3cret", ["Qs3cret"]),
    "passwd": ("mysql+pymysql://db.example.org/cali?passwd=Qs3cret", ["Qs3cret"]),
    "userinfo-and-query": ("postgresql://cali:Us3rinfo@db.example.org/cali?password=Qs3cret", ["Qs3cret", "Us3rinfo"]),
    "encoded-value": ("postgresql://db.example.org/cali?password=Q%40s3cret", ["Q%40s3cret", "Q@s3cret"]),
}


@pytest.mark.parametrize("key", list(QUERY_URLS))
def test_leak_query_string_secrets_in_redaction_helpers(key):
    from cali_address.io.readers import redact_source, _url_secrets

    url, needles = QUERY_URLS[key]
    for text in (redact_url(url), redact_source(url), _short(Boom(f"failed: {url}"), _url_secrets(url))):
        for needle in needles:
            assert needle not in text, f"{needle!r} leaked in {text}"
    assert "db.example.org" in redact_url(url)


@pytest.mark.parametrize("key", list(QUERY_URLS))
def test_leak_query_string_secrets_through_the_exception_chain(key, monkeypatch):
    from cali_address.io import SourceReadError

    url, needles = QUERY_URLS[key]
    _fake_engine_echoing(url, needles[0], monkeypatch)
    with pytest.raises(SourceReadError) as exc:
        read_table(url, table="t")
    text = _chain_text(exc.value)
    for needle in needles:
        assert needle not in text, f"{needle!r} leaked in {text}"


@pytest.mark.parametrize("key", list(QUERY_URLS))
def test_leak_query_string_secrets_in_cli_inspect_json(key, monkeypatch, tmp_path, capsys):
    csv = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    real = read_table
    monkeypatch.setattr(cli, "read_table", lambda source, **k: real(str(csv), **{**k, "fmt": "csv"}))
    url, needles = QUERY_URLS[key]
    assert main(["inspect", url, "--table", "t", "--json"]) == 0
    out = capsys.readouterr()
    for needle in needles:
        assert needle not in out.out + out.err


def test_query_string_redaction_keeps_harmless_parameters():
    masked = redact_url("postgresql://db.example.org/cali?sslmode=require&password=Qs3cret&connect_timeout=5")
    assert "sslmode=require" in masked and "connect_timeout=5" in masked and "Qs3cret" not in masked
