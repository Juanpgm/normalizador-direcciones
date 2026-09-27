"""Artifact-directory resolution shared by training, tuning, evaluation and serving.

An *experiment directory* only has to hold what differs from production
(``model.pt``, ``catastro_emb.pt``, ``tuning.json``, ``reliability.json``,
``train_report.json``, ``experiment.json``). Inputs that do not depend on the
model (the cadastral index, the gazetteer cache, the training tensors and the
negative tables) fall back to the default ``artifacts/`` directory when they are
absent from the experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_ARTIFACTS_DIR",
    "ENV_ARTIFACTS_DIR",
    "SERVING_FILES",
    "SHARED_INPUTS",
    "resolve_artifacts_dir",
    "REQUIRED_ARTIFACTS",
    "OPTIONAL_ARTIFACTS",
    "ArtifactsMissingError",
    "check_artifacts",
    "require_artifacts",
    "shared_path",
    "MODEL_FILES",
    "EXPERIMENT_FILENAME",
    "model_path",
    "sha256_file",
    "sha256_serving_files",
    "code_fingerprint",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
ENV_ARTIFACTS_DIR = "CALI_ARTIFACTS_DIR"

#: The six files the service loads. Experiments must never write these into the
#: default directory; ``deploy/prepare_artifacts.py`` copies them for promotion.
SERVING_FILES = (
    "model.pt",
    "catastro_emb.pt",
    "catastro_docs.parquet",
    "tuning.json",
    "gazetteer.pkl",
    "reliability.json",
)

#: Model-independent inputs that fall back to the default artifacts directory.
SHARED_INPUTS = (
    "catastro_docs.parquet",
    "gazetteer.pkl",
    "train_data.npz",
    "spatial_negatives.npy",
    "plate_negatives.npy",
)


#: Model-dependent files. They never fall back on their own (a retrained experiment that lost its
#: ``model.pt`` must fail loudly); only an experiment that declares ``"inherits_model": true`` in its
#: ``experiment.json`` (a decision-layer-only experiment) reads them from the default directory.
MODEL_FILES = ("model.pt", "catastro_emb.pt")

EXPERIMENT_FILENAME = "experiment.json"


def model_path(name: str, artifacts_dir: str | None = None) -> str:
    """Path of a model file: the experiment dir's own copy, else production when explicitly inherited."""
    if name not in MODEL_FILES:
        raise ValueError(f"{name!r} is not a model file; expected one of {MODEL_FILES}")
    base = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    local = os.path.join(base, name)
    if os.path.exists(local):
        return local
    try:
        with open(os.path.join(base, EXPERIMENT_FILENAME), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return local
    if isinstance(meta, dict) and meta.get("inherits_model") is True:
        return os.path.join(DEFAULT_ARTIFACTS_DIR, name)
    return local


def resolve_artifacts_dir(explicit: str | None = None) -> str:
    """CLI value > ``CALI_ARTIFACTS_DIR`` env > default ``artifacts/``."""
    chosen = explicit or os.environ.get(ENV_ARTIFACTS_DIR) or DEFAULT_ARTIFACTS_DIR
    return os.path.abspath(chosen)


def shared_path(name: str, artifacts_dir: str | None = None) -> str:
    """Path of a shared input: experiment dir if present there, else the default dir."""
    base = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    local = os.path.join(base, name)
    if os.path.exists(local):
        return local
    return os.path.join(DEFAULT_ARTIFACTS_DIR, name)


def is_default_dir(path: str) -> bool:
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(DEFAULT_ARTIFACTS_DIR)


def sha256_file(path: str, chunk: int = 1 << 20) -> str | None:
    """Hex sha256 of a file, or ``None`` when it does not exist."""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_serving_files(artifacts_dir: str | None = None) -> dict:
    base = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    return {name: sha256_file(os.path.join(base, name)) for name in SERVING_FILES}


def code_fingerprint(src_dir: str | None = None) -> str:
    """Git-free fingerprint: sha256 over the sorted ``src/cali_address/*.py`` files."""
    src = src_dir or os.path.join(PROJECT_ROOT, "src", "cali_address")
    h = hashlib.sha256()
    for name in sorted(n for n in os.listdir(src) if n.endswith(".py")):
        h.update(name.encode("utf-8"))
        with open(os.path.join(src, name), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


#: Files ``AddressNormalizer`` cannot start without.
REQUIRED_ARTIFACTS = ("model.pt", "catastro_emb.pt", "catastro_docs.parquet")

#: Files the loader tolerates missing: ``tuning.json`` falls back to the module defaults,
#: ``reliability.json`` leaves the confidence columns empty, ``gazetteer.pkl`` is a cache rebuilt
#: from ``basemaps/``.
OPTIONAL_ARTIFACTS = ("tuning.json", "reliability.json", "gazetteer.pkl")


class ArtifactsMissingError(FileNotFoundError):
    """Required model artifacts are absent; the message lists them and says how to get them."""


def check_artifacts(artifacts_dir: str | None = None) -> tuple[list[str], list[str]]:
    """Return ``(missing_required, missing_optional)`` as the paths the loader would read.

    Resolution mirrors :class:`cali_address.inference.AddressNormalizer` (``model_path`` /
    ``shared_path``), so an experiment directory that inherits the model is judged the same way.
    """
    base = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    resolvers = {"model.pt": model_path, "catastro_emb.pt": model_path, "catastro_docs.parquet": shared_path}
    required = [resolvers[name](name, base) for name in REQUIRED_ARTIFACTS]
    missing = [p for p in required if not os.path.isfile(p)]
    optional = [p for p in (os.path.join(base, name) for name in OPTIONAL_ARTIFACTS) if not os.path.isfile(p)]
    return missing, optional


def require_artifacts(artifacts_dir: str | None = None) -> None:
    """Raise :class:`ArtifactsMissingError` (with an actionable message) unless the required files exist."""
    missing, _ = check_artifacts(artifacts_dir)
    if not missing:
        return
    base = os.path.abspath(artifacts_dir or DEFAULT_ARTIFACTS_DIR)
    listing = "\n".join(f"  - {p}" for p in missing)
    raise ArtifactsMissingError(
        f"model artifacts not found in {base}. Missing required files:\n{listing}\n"
        "Regenerate them by following docs/model-artifacts.md (catastro_emb.pt is not in git), "
        f"or point --artifacts-dir / {ENV_ARTIFACTS_DIR} at a directory that has them."
    )
