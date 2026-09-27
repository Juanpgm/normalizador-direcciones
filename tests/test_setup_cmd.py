"""`cali-address setup`: regenerate catastro_emb.pt from model.pt + catastro_docs.parquet (CPU, tiny fixture)."""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from tiny import tiny_config, tiny_data_dir

import cali_address.bootstrap as bootstrap
import cali_address.paths as paths
from cali_address.cli import main
from cali_address.inference import AddressNormalizer
from cali_address.train import embed_documents, save_checkpoint, train_model


@pytest.fixture(scope="module")
def tiny_art(tmp_path_factory):
    """A complete tiny artifacts dir (model.pt, catastro_emb.pt, catastro_docs.parquet) plus its train data."""
    base = tmp_path_factory.mktemp("tiny_art")
    docs, data, neg = tiny_data_dir(str(base))
    out = train_model(data, neg, tiny_config(), device="cpu")
    save_checkpoint(str(base / "model.pt"), out["model"], out["config"], data)
    emb = embed_documents(out["model"], data["d_tokens"][:, :40], device="cpu").cpu()
    torch.save(emb, str(base / "catastro_emb.pt"))
    return str(base), emb, len(docs)


@pytest.fixture
def art(tmp_path, tiny_art, monkeypatch):
    """A private copy of the tiny artifacts; the repo's real artifacts/ is never consulted."""
    src, _, _ = tiny_art
    dst = tmp_path / "art"
    dst.mkdir()
    for name in ("model.pt", "catastro_emb.pt", "catastro_docs.parquet"):
        shutil.copy(os.path.join(src, name), dst / name)
    empty = tmp_path / "no_default"
    empty.mkdir()
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(empty))
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    return dst


def run_setup(art, capsys, *extra):
    code = main(["setup", "--artifacts-dir", str(art), "--device", "cpu", *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _partials(art):
    return [n for n in os.listdir(art) if n.endswith(".partial") or n.endswith(".tmp")]


# ------------------------------------------------------------------ happy paths
def test_everything_present_is_a_noop_with_self_check(art, capsys):
    before = (art / "catastro_emb.pt").read_bytes()
    code, out, err = run_setup(art, capsys)
    assert code == 0, err
    assert "Already set up" in out and "Setup OK" in out and "rows/s" in out
    assert (art / "catastro_emb.pt").read_bytes() == before


def test_only_embeddings_missing_regenerates_and_loader_accepts_them(art, tiny_art, capsys):
    _, reference, n_docs = tiny_art
    (art / "catastro_emb.pt").unlink()
    code, out, err = run_setup(art, capsys)
    assert code == 0, err
    assert "Setup OK" in out and "Already set up" not in out
    emb = torch.load(art / "catastro_emb.pt", map_location="cpu")
    assert emb.shape == reference.shape == (n_docs, 16)
    assert emb.dtype == reference.dtype == torch.float16
    assert torch.allclose(emb.float(), reference.float(), atol=1e-2)
    norm = AddressNormalizer(str(art), device="cpu")  # its own "docs / embedding mismatch" assertion
    assert norm.doc_emb.shape[0] == len(norm.docs) == n_docs
    assert _partials(art) == []


def test_prints_time_expectation_before_starting(art, capsys):
    (art / "catastro_emb.pt").unlink()
    _, out, _ = run_setup(art, capsys)
    assert out.index("GPU") < out.index("Setup OK")
    assert "CPU" in out and "minutes" in out.lower()


def test_small_batch_size_gives_the_same_embeddings(art, tiny_art, capsys):
    (art / "catastro_emb.pt").unlink()
    code, out, err = run_setup(art, capsys, "--batch-size", "5")
    assert code == 0, err
    emb = torch.load(art / "catastro_emb.pt", map_location="cpu")
    assert torch.allclose(emb.float(), tiny_art[1].float(), atol=1e-2)
    assert "48/48" in out  # progress line reached the end


def test_force_regenerates_an_existing_file(art, tiny_art, capsys):
    torch.save(torch.zeros_like(tiny_art[1]), art / "catastro_emb.pt")
    code, out, err = run_setup(art, capsys, "--force")
    assert code == 0, err
    emb = torch.load(art / "catastro_emb.pt", map_location="cpu")
    assert emb.float().abs().sum() > 0 and torch.allclose(emb.float(), tiny_art[1].float(), atol=1e-2)


def test_optional_files_are_reported_not_required(art, capsys):
    code, out, _ = run_setup(art, capsys)
    assert code == 0
    assert "tuning.json" in out and "optional" in out.lower()


# ------------------------------------------------------------------ missing inputs
def test_model_missing_exit_3_with_actionable_message(art, capsys):
    (art / "model.pt").unlink()
    (art / "catastro_emb.pt").unlink()
    code, out, err = run_setup(art, capsys)
    assert code == 3
    assert "model.pt" in err and "git pull" in err and "docs/model-artifacts.md" in err
    assert not (art / "catastro_emb.pt").exists()


def test_docs_missing_exit_3(art, capsys):
    (art / "catastro_docs.parquet").unlink()
    (art / "catastro_emb.pt").unlink()
    code, out, err = run_setup(art, capsys)
    assert code == 3 and "catastro_docs.parquet" in err and "git" in err
    assert not (art / "catastro_emb.pt").exists()


def test_model_missing_wins_even_with_force(art, capsys):
    (art / "model.pt").unlink()
    code, _, err = run_setup(art, capsys, "--force")
    assert code == 3 and "model.pt" in err


@pytest.mark.parametrize("make", ["absent", "empty"])
def test_artifacts_dir_absent_or_empty(tmp_path, monkeypatch, capsys, make):
    empty_default = tmp_path / "default"
    empty_default.mkdir()
    monkeypatch.setattr(paths, "DEFAULT_ARTIFACTS_DIR", str(empty_default))
    monkeypatch.delenv(paths.ENV_ARTIFACTS_DIR, raising=False)
    target = tmp_path / "nope"
    if make == "empty":
        target.mkdir()
    code, _, err = run_setup(target, capsys)
    assert code == 3 and "model.pt" in err and "Traceback" not in err
    assert target.exists() == (make == "empty")  # setup never creates a missing directory


# ------------------------------------------------------------------ integrity
def test_row_count_mismatch_from_embedding_step_writes_nothing(art, monkeypatch, capsys):
    (art / "catastro_emb.pt").unlink()
    real = bootstrap.embed_documents
    monkeypatch.setattr(bootstrap, "embed_documents", lambda *a, **k: real(*a, **k)[:-1])
    code, _, err = run_setup(art, capsys)
    assert code == 1 and "rows" in err
    assert not (art / "catastro_emb.pt").exists() and _partials(art) == []


def test_stale_existing_embeddings_are_detected_and_force_fixes_them(art, tiny_art, capsys):
    torch.save(tiny_art[1][:-3].clone(), art / "catastro_emb.pt")  # 3 rows short of the docs
    code, out, err = run_setup(art, capsys)
    assert code == 1 and "--force" in err and "Setup OK" not in out
    code, out, err = run_setup(art, capsys, "--force")
    assert code == 0, err
    assert torch.load(art / "catastro_emb.pt", map_location="cpu").shape[0] == tiny_art[2]


def test_corrupt_existing_embeddings_are_reported_not_a_traceback(art, capsys):
    (art / "catastro_emb.pt").write_bytes(b"not a tensor")
    code, out, err = run_setup(art, capsys)
    assert code == 1 and "--force" in err and "Traceback" not in err


def test_interrupted_embedding_leaves_no_file(art, monkeypatch, capsys):
    (art / "catastro_emb.pt").unlink()

    def boom(model, d_tokens, device="cpu", batch=4096, progress=None):
        progress(5, len(d_tokens))
        raise RuntimeError("simulated crash mid-way")

    monkeypatch.setattr(bootstrap, "embed_documents", boom)
    code, _, err = run_setup(art, capsys)
    assert code == 1 and "simulated crash" in err
    assert not (art / "catastro_emb.pt").exists() and _partials(art) == []


def test_interrupted_write_leaves_no_partial_and_keeps_the_old_file(art, monkeypatch, capsys):
    before = (art / "catastro_emb.pt").read_bytes()

    def half_write(obj, path):
        with open(path, "wb") as fh:
            fh.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(bootstrap.torch, "save", half_write)
    code, _, err = run_setup(art, capsys, "--force")
    assert code in (1, 3) and "disk full" in err
    assert (art / "catastro_emb.pt").read_bytes() == before and _partials(art) == []


def test_keyboard_interrupt_mid_way_cleans_up(art, monkeypatch):
    (art / "catastro_emb.pt").unlink()

    def stop(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bootstrap, "embed_documents", stop)
    with pytest.raises(KeyboardInterrupt):
        bootstrap.regenerate_embeddings(str(art), device="cpu")
    assert not (art / "catastro_emb.pt").exists() and _partials(art) == []


# ------------------------------------------------------------------ device / args / self-check
def test_cuda_requested_but_unavailable_falls_back_to_cpu(art, monkeypatch, capsys):
    (art / "catastro_emb.pt").unlink()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(["setup", "--artifacts-dir", str(art), "--device", "cuda"])
    out = capsys.readouterr().out
    assert code == 0 and "device: cpu" in out


def test_default_device_is_cpu_when_cuda_is_absent(art, monkeypatch, capsys):
    (art / "catastro_emb.pt").unlink()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(["setup", "--artifacts-dir", str(art)])
    assert code == 0 and "device: cpu" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["0", "-4", "abc"])
def test_invalid_batch_size_exit_2(art, capsys, bad):
    with pytest.raises(SystemExit) as exc:
        main(["setup", "--artifacts-dir", str(art), "--batch-size", bad])
    assert exc.value.code == 2


def test_self_check_failure_is_reported(art, capsys):
    def broken(artifacts_dir, device):
        raise RuntimeError("model exploded")

    code = main(["setup", "--artifacts-dir", str(art), "--device", "cpu"], normalizer_factory=broken)
    captured = capsys.readouterr()
    assert code == 1 and "Setup FAILED" in captured.err and "model exploded" in captured.err
    assert "Setup OK" not in captured.out


def test_artifacts_dir_taken_from_env_var(art, monkeypatch, capsys):
    monkeypatch.setenv(paths.ENV_ARTIFACTS_DIR, str(art))
    code = main(["setup", "--device", "cpu"])
    assert code == 0 and "Setup OK" in capsys.readouterr().out


def test_embed_documents_reports_progress_and_default_is_unchanged(tiny_art):
    import numpy as np

    from cali_address.train import load_checkpoint

    model, _ = load_checkpoint(os.path.join(tiny_art[0], "model.pt"), "cpu", dropout=0.0)
    tokens = np.zeros((10, 40), dtype=np.int64)
    seen = []
    with_cb = embed_documents(model, tokens, device="cpu", batch=4, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(4, 10), (8, 10), (10, 10)]
    assert torch.equal(with_cb, embed_documents(model, tokens, device="cpu", batch=4))
