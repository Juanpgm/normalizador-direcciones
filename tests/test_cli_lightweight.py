"""`--help`, `inspect` and `formats` must not import torch (fast start, no model stack)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

_SCRIPT = r"""
import sys
from cali_address import cli

def run(argv):
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code
    assert code in (0, None), (argv, code)

for argv in {argvs!r}:
    run(argv)
assert "torch" not in sys.modules, "torch was imported"
print("LIGHT-OK")
"""


def _run(argvs: list[list[str]]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, "-c", _SCRIPT.format(argvs=argvs)], env=env,
                          capture_output=True, text=True, timeout=120)


def _assert_light(argvs: list[list[str]]) -> None:
    proc = _run(argvs)
    assert proc.returncode == 0 and "LIGHT-OK" in proc.stdout, proc.stderr[-2000:] + proc.stdout[-500:]


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "in.csv"
    rows = ["id,direccion,barrio"] + [f"{i},CALLE 5 # {i}-10,X" for i in range(20)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_help_does_not_import_torch():
    _assert_light([["--help"], ["normalize", "--help"], ["inspect", "--help"]])


def test_formats_does_not_import_torch():
    _assert_light([["formats"]])


def test_inspect_csv_does_not_import_torch(csv_file):
    _assert_light([["inspect", csv_file], ["inspect", csv_file, "--json"]])


def test_inspect_semicolon_latin1_csv_does_not_import_torch(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes("codigo;direcci\xf3n\n1;CARRERA 1 # 2-3\n".encode("latin-1"))
    _assert_light([["inspect", str(path)]])


def test_inspect_xlsx_does_not_import_torch(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Reporte"])
    ws.append(["id", "direccion"])
    ws.append([1, "CALLE 5 # 10-20"])
    path = tmp_path / "x.xlsx"
    wb.save(path)
    _assert_light([["inspect", str(path)]])


def test_inspect_empty_csv_error_path_does_not_import_torch(tmp_path):
    """The user-error path (exit code != 0) must stay light too."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=SRC)
    script = ("import sys\nfrom cali_address import cli\n"
              f"code = cli.main(['inspect', {str(path)!r}])\n"
              "assert code != 0\nassert 'torch' not in sys.modules\nprint('LIGHT-OK')\n")
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert "LIGHT-OK" in proc.stdout, proc.stderr[-2000:]
