# PlasFlow v2 — multi-stage Docker build
# Base: debian:bookworm-slim  (~80 MB)
# Final image target: < 3 GB (excluding database volumes)

# ---------------------------------------------------------------------------
# Stage 1: build environment (installs build tools + Python deps)
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        build-essential curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch (GPU not available in Docker by default)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# Copy and install plasflow2
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps -e .

# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime system deps: DIAMOND, BLAST+, Prodigal
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        diamond-aligner \
        ncbi-blast+ \
        prodigal \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src /opt/plasflow2/src

# MOB-suite is conda-only; run on CPU machine via subprocess or pre-installed env.
# To include: add a miniconda layer or use the bioconda image as base.

# Create data directory for database mounts
RUN mkdir -p /data/databases /data/models /data/test

WORKDIR /work

ENTRYPOINT ["plasflow2"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Usage:
#   docker build -t plasflow2 -f docker/Dockerfile .
#
#   # Mount databases and run full pipeline
#   docker run --rm \
#     -v /path/to/databases:/data/databases:ro \
#     -v $(pwd)/results:/work/results \
#     plasflow2 run \
#       --input /work/assembly.fasta \
#       --output /work/results \
#       --context clinical
# ---------------------------------------------------------------------------
