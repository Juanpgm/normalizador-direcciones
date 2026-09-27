"""CLI: exit code 4 (every row failed), overwrite semantics and --config path resolution."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from dataset_helpers import ADDR_OK, ADDR_RAISES, CADASTRE, exploding_stub, stub, write_csv

from cali_address.cli import main
from cali_address.io.config import resolve_config_paths


def run(argv, *, normalizer=None):
    made = []

    def factory(artifacts_dir, device):
        made.append((artifacts_dir, device))
        return normalizer or stub()

    code = main(argv, normalizer_factory=factory, gazetteer_loader=lambda *a, **k: None)
    return code, made


def _read_out(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# exit code 4: the run finished but EVERY row failed
# ---------------------------------------------------------------------------
class _AlwaysBroken(type(stub())):
    def score_components(self, queries, k=20):
        raise RuntimeError("model is broken")


def _broken():
    return _AlwaysBroken(CADASTRE)


def test_every_row_error_exits_4_and_still_writes_output_and_summary(tmp_path, capsys):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_OK, ADDR_OK]}))
    out, sj = tmp_path / "o.csv", tmp_path / "s.json"
    code, _ = run(["normalize", src, "-o", str(out), "--summary-json", str(sj)], normalizer=_broken())
    err = capsys.readouterr().err
    assert code == 4 and "error:" in err and "every row" in err and "Traceback" not in err
    assert _read_out(out)["estado"].tolist() == ["ERROR"] * 3
    assert json.loads(sj.read_text(encoding="utf-8"))["error"] == 3


def test_single_row_error_exits_4(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    assert run(["normalize", src, "-o", str(tmp_path / "o.csv")], normalizer=_broken())[0] == 4


def test_mixed_errors_still_exit_0(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_RAISES, ADDR_OK]}))
    assert run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--chunk-size", "1"],
               normalizer=exploding_stub())[0] == 0


def test_zero_rows_still_exit_0(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("id,direccion\n", encoding="utf-8")
    assert run(["normalize", str(p), "-o", str(tmp_path / "o.csv")], normalizer=_broken())[0] == 0


def test_all_rows_out_of_area_is_not_an_error_exit(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK], "mun": ["Bogota"]}))
    code, _ = run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--municipality-col", "mun"], normalizer=_broken())
    assert code == 0


def test_all_error_dry_run_exits_4_too(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    assert run(["normalize", src, "--dry-run", "5"], normalizer=_broken())[0] == 4


def test_output_is_overwritten_atomically_even_when_it_is_the_input_path(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    assert run(["normalize", src, "-o", src])[0] == 0
    assert _read_out(src)["estado"].tolist() == ["OK"]
    assert [f for f in os.listdir(tmp_path) if f.endswith(".partial")] == []


def test_existing_output_is_replaced_not_appended(tmp_path):
    src = write_csv(tmp_path / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    out = tmp_path / "o.csv"
    out.write_text("stale,content\n1,2\n", encoding="utf-8")
    assert run(["normalize", src, "-o", str(out)])[0] == 0
    assert "stale" not in out.read_text(encoding="utf-8-sig") and len(_read_out(out)) == 1


# ---------------------------------------------------------------------------
# --config: relative paths resolve against the config file's directory
# ---------------------------------------------------------------------------
def _cfg_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    write_csv(proj / "data" / "in.csv", pd.DataFrame({"direccion": [ADDR_OK]}))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    return proj, cwd


def test_config_relative_paths_resolve_against_the_config_directory(tmp_path, monkeypatch):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "data/in.csv"\noutput = "out/o.csv"\nsummary_json = "s.json"\n',
                                   encoding="utf-8")
    monkeypatch.chdir(cwd)
    assert run(["normalize", "--config", str(proj / "run.toml")])[0] == 0
    assert (proj / "out" / "o.csv").exists() and (proj / "s.json").exists()
    assert os.listdir(cwd) == []  # nothing was written relative to the CWD


def test_config_relative_paths_work_with_a_relative_config_argument(tmp_path, monkeypatch):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "data/in.csv"\noutput = "o.csv"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert run(["normalize", "--config", os.path.join("proj", "run.toml")])[0] == 0
    assert (proj / "o.csv").exists()


def test_config_absolute_paths_are_unchanged(tmp_path, monkeypatch):
    proj, cwd = _cfg_project(tmp_path)
    target = tmp_path / "abs_out.csv"
    (proj / "run.toml").write_text(
        f'input = "{(proj / "data" / "in.csv").as_posix()}"\noutput = "{target.as_posix()}"\n', encoding="utf-8")
    monkeypatch.chdir(cwd)
    assert run(["normalize", "--config", str(proj / "run.toml")])[0] == 0 and target.exists()


def test_command_line_paths_stay_relative_to_the_cwd_and_beat_the_config(tmp_path, monkeypatch):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "data/in.csv"\noutput = "from_cfg.csv"\n', encoding="utf-8")
    write_csv(cwd / "mine.csv", pd.DataFrame({"direccion": [ADDR_OK, ADDR_OK]}))
    monkeypatch.chdir(cwd)
    assert run(["normalize", "mine.csv", "-o", "mine_out.csv", "--config", str(proj / "run.toml")])[0] == 0
    assert len(_read_out(cwd / "mine_out.csv")) == 2 and not (proj / "from_cfg.csv").exists()


def test_config_url_inputs_are_left_untouched(tmp_path, monkeypatch, capsys):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "sqlite:///rel.db"\noutput = "o.csv"\ntable = "t"\n', encoding="utf-8")
    monkeypatch.chdir(cwd)
    code, _ = run(["normalize", "--config", str(proj / "run.toml")])
    err = capsys.readouterr().err
    assert code == 3 and "rel.db" in err and str(proj) not in err


def test_config_missing_relative_input_names_the_resolved_path(tmp_path, monkeypatch, capsys):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "data/missing.csv"\noutput = "o.csv"\n', encoding="utf-8")
    monkeypatch.chdir(cwd)
    code, _ = run(["normalize", "--config", str(proj / "run.toml")])
    err = capsys.readouterr().err
    assert code == 3 and "missing.csv" in err and str(proj) in err


def test_config_artifacts_dir_is_resolved_too(tmp_path, monkeypatch):
    proj, cwd = _cfg_project(tmp_path)
    (proj / "run.toml").write_text('input = "data/in.csv"\noutput = "o.csv"\nartifacts_dir = "arts"\n', encoding="utf-8")
    monkeypatch.chdir(cwd)
    code, made = run(["normalize", "--config", str(proj / "run.toml")])
    assert code == 0 and made[0][0] == str(proj / "arts")


# ---------------------------------------------------------------------------
# resolve_config_paths
# ---------------------------------------------------------------------------
def test_resolve_config_paths_only_touches_path_options(tmp_path):
    base = str(tmp_path / "cfgdir")
    out = resolve_config_paths(
        {"input": "in.csv", "output": "o/out.csv", "summary_json": "s.json", "artifacts_dir": "a", "basemaps": "b",
         "sheet": "in.csv", "table": "in.csv", "chunk_size": 5}, base)
    assert out["input"] == os.path.join(base, "in.csv") and out["output"] == os.path.join(base, "o", "out.csv")
    assert out["summary_json"] == os.path.join(base, "s.json")
    assert out["artifacts_dir"] == os.path.join(base, "a") and out["basemaps"] == os.path.join(base, "b")
    assert out["sheet"] == "in.csv" and out["table"] == "in.csv" and out["chunk_size"] == 5


@pytest.mark.parametrize("value", [
    "sqlite:///rel.db", "postgresql://u:p@h/db", "https://example.org/x.csv", "",
])
def test_resolve_config_paths_leaves_urls_and_empty_untouched(tmp_path, value):
    assert resolve_config_paths({"input": value}, str(tmp_path)) == {"input": value}


def test_resolve_config_paths_keeps_absolute_paths(tmp_path):
    absolute = str(tmp_path / "abs.csv")
    assert resolve_config_paths({"input": absolute}, str(tmp_path / "elsewhere")) == {"input": absolute}


def test_resolve_config_paths_does_not_mutate_its_argument(tmp_path):
    original = {"input": "in.csv"}
    resolve_config_paths(original, str(tmp_path))
    assert original == {"input": "in.csv"}
