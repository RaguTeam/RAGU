# RAGU search service.
#
# Build from the repository root:
#   docker build -t ragu-api .
#
# The image serves a prebuilt graph; it does not build one. Mount the storage
# folder produced by a build run at /data/graph.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

# Present for source-only dependencies; the locked set installs from wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY ragu/ ./ragu/

ARG RAGU_SYNC_EXTRAS=""
RUN uv sync --frozen --no-dev --no-editable --extra api ${RAGU_SYNC_EXTRAS}

ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/tiktoken \
 && /opt/venv/bin/python -c "import tiktoken; [tiktoken.get_encoding(name) for name in ('cl100k_base', 'o200k_base')]"

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    RAGU_API_STORAGE_FOLDER=/data/graph

# curl is used by the healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken

RUN groupadd --gid 1000 ragu \
 && useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/false --system ragu

# The service is part of the installed package (ragu.api), so nothing but the
# venv needs copying.
WORKDIR /app
RUN mkdir -p /data/graph /data/cache && chown -R ragu:ragu /data

USER ragu:ragu

EXPOSE 8020

# /health/ready answers 503 until the graph is loaded, so -f is enough; the
# 10-minute start period covers loading a large graph.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fs http://localhost:8020/health/ready || exit 1

ENTRYPOINT ["python", "-m", "ragu.api"]
CMD ["--host", "0.0.0.0", "--port", "8020"]
