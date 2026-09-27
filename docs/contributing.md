# Contributing

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,api,sql,excel,geo]"
```

Python 3.11+. The model artifacts are optional for development: see
[model-artifacts.md](model-artifacts.md) if you need the real model.

## Tests

```bash
python -m pytest -q
```

- The suite must stay green without the artifacts: tests that need the real model carry
  `@pytest.mark.requires_artifacts` (skipped automatically when `catastro_emb.pt` is missing).
- Write the failing test first, and cover edge cases, not only the happy path: boundary values, empty / null /
  malformed input, failures (I/O, network, server errors) and invalid state transitions.
- CI (`.github/workflows/ci.yml`) runs the same command on Linux with CPU-only torch.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`chore:`, `ci:`. Keep each commit to one reviewable unit with its tests and docs.

## Privacy rules

- Never commit `context/`, `outputs/` or `artifacts/splits/`: they hold real citizen data and raw evaluation
  addresses. Git history keeps them forever, so a later delete does not undo a leak.
- Never commit large binaries (`catastro_emb.pt`, `*.npz`, `*.npy`) or notebooks.
- Do not paste real addresses or personal data into tests, docs or issues; use synthetic ones.

## Experiments and splits

Retraining and decision-layer variants run in isolated directories under `artifacts/experiments/<tag>/` and never
touch production. Selection uses the `dev` split only; the `frozen_test` split is for the final measurement.
See [experiments.md](experiments.md) for the workflow and [splits.md](splits.md) for the rules and rationale.
