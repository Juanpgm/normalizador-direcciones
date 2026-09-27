"""`normalize` must fail early and clearly when the model artifacts are missing."""

from __future__ import annotations

import os

import pytest

from cali_address.cli import main
from cali_address.paths import (
    ENV_ARTIFACTS_DIR,
    MODEL_FILES,
    REQUIRED_ARTIFACTS,
    ArtifactsMissingError,
    check_artifacts,
    require_artifacts,
)


@pytest.fixture
def src(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("id,direccion\n1,CALLE 5 # 10-20\n", encoding="utf-8")
    return str(path)


def _touch(directory, *names):
    os.makedirs(directory, exist_ok=True)
    for name in names:
        with open(os.path.join(directory, name), "wb") as fh:
            fh.write(b"x")


def _run(argv, capsys, monkeypatch):
    monkeypatch.delenv(ENV_ARTIFACTS_DIR, raising=False)
    code = main(argv)
    return code, capsys.readouterr().err


def test_missing_directory_lists_every_required_file(tmp_path, src, capsys, monkeypatch):
    out = tmp_path / "o.csv"
    code, err = _run(["normalize", src, "-o", str(out), "--artifacts-dir", str(tmp_path / "nope")], capsys, monkeypatch)
    assert code == 3 and "Traceback" not in err
    # catastro_docs.parquet falls back to the repo's artifacts/ (tracked in git), so it is not always listed.
    for name in MODEL_FILES:
        assert name in err
    assert "docs/model-artifacts.md" in err
    assert not out.exists()


def test_empty_directory(tmp_path, src, capsys, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, err = _run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--artifacts-dir", str(empty)], capsys, monkeypatch)
    assert code == 3
    assert all(name in err for name in MODEL_FILES)


def test_only_the_missing_file_is_reported(tmp_path, src, capsys, monkeypatch):
    d = tmp_path / "art"
    _touch(d, "model.pt", "catastro_docs.parquet")
    code, err = _run(["normalize", src, "-o", str(tmp_path / "o.csv"), "--artifacts-dir", str(d)], capsys, monkeypatch)
    assert code == 3
    assert "catastro_emb.pt" in err
    assert "model.pt" not in err.replace("docs/model-artifacts.md", "")
    assert "catastro_docs.parquet" not in err


def test_dry_run_also_checks(tmp_path, src, capsys, monkeypatch):
    code, err = _run(["normalize", src, "--dry-run", "3", "--artifacts-dir", str(tmp_path / "nope")], capsys, monkeypatch)
    assert code == 3 and "model.pt" in err


def test_env_var_is_honoured_as_default(tmp_path, src, capsys, monkeypatch):
    target = tmp_path / "from-env"
    monkeypatch.setenv(ENV_ARTIFACTS_DIR, str(target))
    code = main(["normalize", src, "-o", str(tmp_path / "o.csv")])
    err = capsys.readouterr().err
    assert code == 3 and str(target) in err


def test_flag_wins_over_env(tmp_path, src, capsys, monkeypatch):
    monkeypatch.setenv(ENV_ARTIFACTS_DIR, str(tmp_path / "from-env"))
    code = main(["normalize", src, "-o", str(tmp_path / "o.csv"), "--artifacts-dir", str(tmp_path / "from-flag")])
    err = capsys.readouterr().err
    assert code == 3 and "from-flag" in err and "from-env" not in err


def test_check_artifacts_separates_required_from_optional(tmp_path):
    d = tmp_path / "a"
    _touch(d, *REQUIRED_ARTIFACTS)
    required, optional = check_artifacts(str(d))
    assert required == []
    assert sorted(os.path.basename(p) for p in optional) == ["gazetteer.pkl", "reliability.json", "tuning.json"]
    require_artifacts(str(d))  # optional files never raise


def test_a_directory_named_like_a_file_does_not_count(tmp_path):
    d = tmp_path / "a"
    _touch(d, "model.pt", "catastro_docs.parquet")
    os.makedirs(d / "catastro_emb.pt")
    required, _ = check_artifacts(str(d))
    assert [os.path.basename(p) for p in required] == ["catastro_emb.pt"]


def test_malformed_experiment_json_is_ignored(tmp_path):
    d = tmp_path / "a"
    _touch(d)
    (d / "experiment.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactsMissingError) as info:
        require_artifacts(str(d))
    assert isinstance(info.value, FileNotFoundError)
    assert "model.pt" in str(info.value)


def test_error_message_mentions_the_env_var_and_flag(tmp_path):
    with pytest.raises(ArtifactsMissingError) as info:
        require_artifacts(str(tmp_path / "x"))
    text = str(info.value)
    assert "--artifacts-dir" in text and ENV_ARTIFACTS_DIR in text
