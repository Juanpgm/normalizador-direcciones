"""Copy only the artifacts the HTTP service actually needs into deploy/artifacts/.

``artifacts/`` is ~415 MB: training tensors, evaluation parquets, the IDESC
geocoder cache, sweep CSVs, logs. Serving needs six files and nothing else, so
the Docker image copies this curated directory instead. Run it before building:

    python deploy/prepare_artifacts.py

Idempotent: a file whose size and mtime already match is left alone. Prints the
total size copied so a deploy cannot silently start shipping a 200 MB surprise.

Privacy: this script only ever reads from ``artifacts/``. It never touches
``context/`` or ``outputs/``, which hold real citizen data and must never enter
a container image or a git repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

#: The only artifacts ``cali_address.service`` / ``cali_address.api`` load.
SERVING_ARTIFACTS = (
    "model.pt",             # char-Transformer encoder checkpoint
    "catastro_emb.pt",      # 330k cadastral embeddings
    "catastro_docs.parquet",  # the cadastral records behind those embeddings
    "tuning.json",          # calibrated fuse weights + confidence threshold
    "gazetteer.pkl",        # parsed IDESC basemaps cache
    "reliability.json",     # calibrated reliability model (tiny; scripts/fit_reliability.py)
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_TARGET = os.path.join(PROJECT_ROOT, "deploy", "artifacts")


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _is_current(source: str, target: str) -> bool:
    if not os.path.exists(target):
        return False
    src, dst = os.stat(source), os.stat(target)
    return src.st_size == dst.st_size and int(src.st_mtime) <= int(dst.st_mtime)


def prepare(source_dir: str = DEFAULT_SOURCE, target_dir: str = DEFAULT_TARGET) -> int:
    """Copy the serving artifacts; return the number of bytes now in ``target_dir``."""
    missing = [n for n in SERVING_ARTIFACTS if not os.path.exists(os.path.join(source_dir, n))]
    if missing:
        raise SystemExit(
            f"error: missing artifact(s) in {source_dir}: {', '.join(missing)}\n"
            "Run the training / tuning pipeline first, or point --source at a directory "
            "that has them."
        )
    os.makedirs(target_dir, exist_ok=True)
    total = 0
    for name in SERVING_ARTIFACTS:
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        size = os.path.getsize(src)
        total += size
        if _is_current(src, dst):
            print(f"  up to date  {name:24} {_human(size):>10}")
            continue
        print(f"  copying     {name:24} {_human(size):>10}")
        shutil.copy2(src, dst)
    extra = [
        n for n in sorted(os.listdir(target_dir))
        if n not in SERVING_ARTIFACTS and os.path.isfile(os.path.join(target_dir, n))
    ]
    if extra:
        print(f"  note: {len(extra)} unexpected file(s) in {target_dir}: {', '.join(extra)}")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="directory holding the full artifacts")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="curated directory the image copies")
    args = ap.parse_args(argv)
    print(f"preparing serving artifacts: {args.source} -> {args.target}")
    total = prepare(args.source, args.target)
    print(f"total: {_human(total)} across {len(SERVING_ARTIFACTS)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
