"""`cali-address web`: preflight checks and hand-off to uvicorn (mocked: no real server here)."""

from __future__ import annotations

import importlib
import os
import socket

import pytest

import cali_address.cli as cli
import cali_address.paths as paths
from cali_address.cli import main


@pytest.fixture(autouse=True)
def _restore_api_env(monkeypatch):
    """`web` exports env vars for the API; make monkeypatch undo them after every test."""
    for name in ("ARTIFACTS_DIR", "NORMALIZER_DEVICE"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


@pytest.fixture
def art(tmp_path, monkeypatch):
    d = tmp_path / "art"
    d.mkdir()
    for name in paths.REQUIRED_ARTIFACTS:
        (d / name).write_bytes(b"x")
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(tmp_path / "no_default"))
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("NORMALIZER_DEVICE", raising=False)
    return d


@pytest.fixture
def calls(monkeypatch):
    uv = importlib.import_module("uvicorn")
    made = {"run": [], "browser": []}
    monkeypatch.setattr(uv, "run", lambda *a, **k: made["run"].append((a, k)))
    monkeypatch.setattr(cli, "_schedule_browser", lambda url, delay=1.5: made["browser"].append(url))
    return made


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_starts_the_existing_app_and_exports_the_api_env_var(art, calls, capsys):
    port = free_port()
    assert main(["web", "--artifacts-dir", str(art), "--port", str(port)]) == 0
    (args, kwargs), = calls["run"]
    assert args == ("cali_address.api.main:app",) and kwargs == {"host": "127.0.0.1", "port": port}
    assert os.environ["ARTIFACTS_DIR"] == str(art)  # the API's variable, not CALI_ARTIFACTS_DIR
    assert calls["browser"] == [f"http://127.0.0.1:{port}"]
    assert f"http://127.0.0.1:{port}" in capsys.readouterr().out


def test_app_import_path_resolves():
    module = importlib.import_module("cali_address.api.main")
    assert hasattr(module, "app")
    assert module.artifacts_dir.__name__ == "artifacts_dir"


def test_no_browser_flag(art, calls):
    assert main(["web", "--artifacts-dir", str(art), "--port", str(free_port()), "--no-browser"]) == 0
    assert calls["browser"] == [] and len(calls["run"]) == 1


def test_env_var_selects_artifacts_dir(art, calls, monkeypatch):
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(art))
    assert main(["web", "--port", str(free_port()), "--no-browser"]) == 0
    assert os.environ["ARTIFACTS_DIR"] == str(art)


def test_device_is_forwarded_only_when_given(art, calls):
    main(["web", "--artifacts-dir", str(art), "--port", str(free_port()), "--no-browser"])
    assert "NORMALIZER_DEVICE" not in os.environ
    main(["web", "--artifacts-dir", str(art), "--port", str(free_port()), "--no-browser", "--device", "cuda"])
    assert os.environ["NORMALIZER_DEVICE"] == "cuda"


def test_wildcard_host_shows_localhost(art, calls, capsys):
    port = free_port()
    assert main(["web", "--artifacts-dir", str(art), "--port", str(port), "--host", "0.0.0.0", "--no-browser"]) == 0
    assert f"http://localhost:{port}" in capsys.readouterr().out


def test_missing_extras_exit_3(art, calls, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_missing_web_extras", lambda: ["uvicorn"])
    assert main(["web", "--artifacts-dir", str(art), "--no-browser"]) == 3
    assert 'pip install -e ".[api]"' in capsys.readouterr().err
    assert calls["run"] == []


def test_real_missing_import_is_detected(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", None)  # `import uvicorn` now raises ImportError
    assert cli._missing_web_extras() == ["uvicorn"]


def test_port_in_use_exit_3(art, calls, capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert main(["web", "--artifacts-dir", str(art), "--port", str(port), "--no-browser"]) == 3
    err = capsys.readouterr().err
    assert str(port) in err and "--port" in err and "Traceback" not in err
    assert calls["run"] == []


def test_probe_socket_is_released_before_serving(art, calls):
    port = free_port()
    main(["web", "--artifacts-dir", str(art), "--port", str(port), "--no-browser"])
    with socket.socket() as again:  # would fail if the preflight probe still held the port
        again.bind(("127.0.0.1", port))


@pytest.mark.parametrize("bad", ["0", "-1", "70000", "65536", "abc", "", "8000.5"])
def test_invalid_port_exit_2(art, calls, bad):
    with pytest.raises(SystemExit) as exc:
        main(["web", "--artifacts-dir", str(art), "--port", bad])
    assert exc.value.code == 2 and calls["run"] == []


@pytest.mark.parametrize("port", ["1", "65535"])
def test_boundary_ports_are_accepted_by_the_parser(port):
    assert cli.build_parser().parse_args(["web", "--port", port]).port == int(port)


def test_artifacts_missing_exit_3_and_points_to_setup(tmp_path, calls, monkeypatch, capsys):
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(tmp_path / "no_default"))
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    (tmp_path / "empty").mkdir()
    assert main(["web", "--artifacts-dir", str(tmp_path / "empty"), "--port", str(free_port()), "--no-browser"]) == 3
    err = capsys.readouterr().err
    assert "model.pt" in err and "setup" in err
    assert calls["run"] == [] and "ARTIFACTS_DIR" not in os.environ


def test_schedule_browser_opens_the_url(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url))

    class Immediate:
        def __init__(self, delay, fn, args=()):
            self.fn, self.args, self.daemon = fn, args, False

        def start(self):
            self.fn(*self.args)

    monkeypatch.setattr(cli.threading, "Timer", Immediate)
    cli._schedule_browser("http://127.0.0.1:1")
    assert opened == ["http://127.0.0.1:1"]
