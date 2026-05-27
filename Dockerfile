# PlasFlow v2 — Multi-stage Docker image
#
# Stage 1 (builder): install Python deps via Poetry into a venv
# Stage 2 (runtime): copy venv + install system tools (DIAMOND, MOB-suite)
#
# Usage:
#   docker build -t plasflow2 .
#   docker run --rm \
#     -v /path/to/data:/data \
#     -v /path/to/results:/results \
#     plasflow2 run \
#       --input   /data/assembly.fasta \
#       --output  /results/ \
#       --card-db /data/databases/card/card.dmnd \
#       --aro-index /data/databases/card/aro_index.tsv \
#       --threads 8
#
# To print the setup guide inside the container:
#   docker run --rm plasflow2 setup

# ─── Stage 1: Python dependency builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (no virtualenvs inside container — we'll copy the venv out)
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy only dependency files first (layer-cache friendly)
COPY pyproject.toml poetry.lock* ./

# Install runtime dependencies (no dev extras)
RUN poetry install --only main --no-root

# Copy source and install the package itself
COPY src/ src/
RUN poetry install --only main


# ─── Stage 2: Runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Raza <shahbaz.invincible3182@gmail.com>"
LABEL description="PlasFlow v2 — metagenomic contig classifier and AMR risk scorer"
LABEL org.opencontainers.image.source="https://github.com/Raza-pl/plasflow2.0"

# System packages:
#   - DIAMOND (bioinformatics aligner)
#   - mob_suite (plasmid mobility typing) — installed via pip inside conda;
#     here we install its Python deps and the binary separately
#   - libgomp1: OpenMP for DIAMOND multithreading
#   - wget / curl: for plasflow2 setup downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        wget \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

# ── Install DIAMOND ───────────────────────────────────────────────────────────
ARG DIAMOND_VERSION=2.1.9
RUN wget -q "https://github.com/bbuchfink/diamond/releases/download/v${DIAMOND_VERSION}/diamond-linux64.tar.gz" \
    -O /tmp/diamond.tar.gz \
    && tar -xzf /tmp/diamond.tar.gz -C /usr/local/bin diamond \
    && rm /tmp/diamond.tar.gz \
    && diamond --version

# ── Install MOB-suite via pip ────────────────────────────────────────────────
# MOB-suite has a pip package but requires some system dependencies
RUN pip install --no-cache-dir mob_suite==3.1.9

# ── Copy Python venv from builder ────────────────────────────────────────────
COPY --from=builder /build/.venv /opt/plasflow2-venv
ENV PATH="/opt/plasflow2-venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/plasflow2-venv"

# ── Copy application source ──────────────────────────────────────────────────
WORKDIR /app
COPY src/ src/
COPY data/models/ data/models/

# ── Runtime defaults ─────────────────────────────────────────────────────────
# Volumes the user should mount:
#   /data/databases/  — CARD + GTDB databases (read-only)
#   /data/input/      — input FASTA files (read-only)
#   /results/         — output directory (read-write)
VOLUME ["/data", "/results"]

# Make plasflow2 accessible
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["plasflow2"]
CMD ["--help"]
