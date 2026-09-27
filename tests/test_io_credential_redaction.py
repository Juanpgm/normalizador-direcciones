"""No credential in a source URL ever reaches stdout, stderr, --summary-json or an exception chain."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from dataset_helpers import stub

from cali_address.cli import main
from cali_address.io import DatasetError, read_table
from cali_address.io.readers import infer_format, redact_source, redact_url

SECRET = "s3cretpw"
ENCODED_SECRET = "p%40ss"
DECODED_SECRET = "p@ss"

BASES = {
    "userinfo": ("https://bob:s3cretpw@example.invalid/export", [SECRET]),
    "token-param": ("https://example.invalid/export?token=s3cretpw", [SECRET]),
    "api-key-param": ("https://example.invalid/export?api_key=s3cretpw&x=1", [SECRET]),
    "encoded-userinfo": ("https://bob:p%40ss@example.invalid/export", [ENCODED_SECRET, DECODED_SECRET]),
}
EXTENSIONS = ["", ".csv", ".xlsx", ".unknownext"]
# with a query string the extension goes on the path, before the "?"
CASES = [
    pytest.param(
        (base.replace("/export", "/export" + ext, 1) if ext else base), secrets, ext in ("", ".unknownext"),
        id=f"{name}{ext or '-noext'}",
    )
    for name, (base, secrets) in BASES.items()
    for ext in EXTENSIONS
]


@pytest.fixture(autouse=True)
def _network_echoes_the_url(monkeypatch):
    """A network stack that repeats the full URL in its error, the worst case for a leak."""
    def fake_urlopen(url, *a, **k):
        raise urllib.error.URLError(f"unreachable: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _chain(exc):
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        yield exc
        yield from _chain(exc.__cause__)
        exc = exc.__context__


def _assert_clean(text, secrets):
    for secret in secrets:
        assert secret not in text, f"{secret!r} leaked in: {text}"


def _run(argv, tmp_path, secrets, capsys):
    def factory(artifacts_dir, device):
        return stub()

    code = main(argv, normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None)
    out = capsys.readouterr()
    _assert_clean(out.out + out.err, secrets)
    return code


@pytest.mark.parametrize(("url", "secrets", "is_format_error"), CASES)
def test_inspect_never_prints_the_secret(url, secrets, is_format_error, tmp_path, capsys):
    code = _run(["inspect", url, "--json"], tmp_path, secrets, capsys)
    assert code == (2 if is_format_error else 3)


@pytest.mark.parametrize(("url", "secrets", "is_format_error"), CASES)
def test_normalize_never_prints_the_secret(url, secrets, is_format_error, tmp_path, capsys):
    summary = tmp_path / "summary.json"
    out = tmp_path / "o.csv"
    code = _run(["normalize", url, "-o", str(out), "--summary-json", str(summary)], tmp_path, secrets, capsys)
    assert code == (2 if is_format_error else 3)
    assert not out.exists()
    if summary.exists():
        _assert_clean(summary.read_text(encoding="utf-8"), secrets)


@pytest.mark.parametrize(("url", "secrets", "is_format_error"), CASES)
def test_read_table_and_infer_format_exception_chains_are_clean(url, secrets, is_format_error):
    with pytest.raises(DatasetError) as exc:
        read_table(url)
    for err in _chain(exc.value):
        _assert_clean(str(err) + repr(err) + repr(err.args), secrets)
    if is_format_error:
        with pytest.raises(DatasetError) as exc:
            infer_format(url)
        _assert_clean(str(exc.value) + repr(exc.value.args), secrets)


@pytest.mark.parametrize("command", ["inspect", "normalize"])
@pytest.mark.parametrize("url", [
    "postgresql://bob:s3cretpw@db.invalid/x", "https://bob:s3cretpw@example.invalid/export?x=1",
    "postgresql://bob:x@db.invalid/x?password=s3cretpw",
])
def test_forcing_a_file_format_on_a_url_does_not_leak(command, url, tmp_path, capsys):
    """--format csv on a DB URL takes the local-file path: 'input file not found: <url>' must be redacted."""
    argv = [command, url, "--format", "csv"] + (["-o", str(tmp_path / "o.csv")] if command == "normalize" else [])
    assert _run(argv, tmp_path, [SECRET], capsys) == 3


@pytest.mark.parametrize("param", [
    "token", "password", "sslpassword", "passwd", "api_key", "apikey", "key", "secret", "access_token", "sig",
    "signature", "auth",
])
def test_every_documented_secret_param_is_masked(param):
    url = f"https://example.invalid/export?{param}=s3cretpw&x=1"
    assert SECRET not in redact_source(url)
    assert SECRET not in redact_url(url)
    assert "x=1" in redact_source(url)


def test_harmless_params_are_left_alone():
    assert "page=2" in redact_source("https://example.invalid/export?page=2")
