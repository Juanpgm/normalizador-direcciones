# Serving image for the Cali address normalizer HTTP API.
#
# Build prerequisite (the image does NOT copy the 415 MB artifacts/ directory):
#
#     python deploy/prepare_artifacts.py
#     docker build -t cali-normalizador .
#
# What is deliberately NOT in this image: context/ and outputs/ (real citizen
# data: insured-property registry, disaster-victim registry with names and ID
# numbers, field inspection reports), the raw artifacts/ directory, .git,
# scripts/, tests/ and direcciones_ANN.ipynb. See .dockerignore.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    NORMALIZER_DEVICE=cpu \
    ARTIFACTS_DIR=/app/artifacts \
    BASEMAPS_DIR=/app/basemaps \
    DOCS_DIR=/app/docs

WORKDIR /app

# shapely, pyproj and pyarrow ship manylinux wheels with GEOS/PROJ/Arrow bundled,
# so no apt build or runtime packages are needed on top of python:3.12-slim.
# curl is installed only so the image can health-check itself.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, in its own layer and from its own index so the pin cannot leak
# into the resolution of everything else.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docs/ ./docs/
COPY basemaps/ ./basemaps/
# Curated 199 MB of serving artifacts produced by deploy/prepare_artifacts.py,
# never the full artifacts/ directory.
COPY deploy/artifacts/ ./artifacts/

RUN python -c "import cali_address.service, cali_address.api.main" \
    && test -f /app/artifacts/model.pt \
    && test -f /app/artifacts/catastro_emb.pt \
    && test -f /app/artifacts/catastro_docs.parquet \
    && test -f /app/artifacts/tuning.json \
    && test -f /app/artifacts/reliability.json \
    && test -f /app/artifacts/gazetteer.pkl \
    && test -f /app/basemaps/barrios_veredas.geojson

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Shell form so Railway's $PORT is expanded at runtime.
CMD uvicorn cali_address.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
