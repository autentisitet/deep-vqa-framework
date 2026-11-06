# Dockerfile - Multi-stage build

# =============================================
# Stage 1: Base Environment (Shared Layer)
# ===============================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock requirements.txt ./
# Keep README.md: pyproject.toml declares readme = "README.md", and Hatchling
# needs it when uv builds/installs the local project during setup.
# Keep LICENSE available in the image for package/license metadata and distribution compliance.
COPY README.md LICENSE ./

COPY scripts/ ./scripts/

ARG BOOTSTRAP_ARGS="--mirror"
ARG SETUP_ARGS="--mirror"
ARG INSTALL_DEV=false
ARG USE_BUILD_PROXY=false

# Install system and Python dependencies
RUN if [ "$USE_BUILD_PROXY" != "true" ]; then \
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
    fi && \
    bash scripts/bootstrap.sh ${BOOTSTRAP_ARGS} && \
    setup_args="${SETUP_ARGS}" && \
    if [ "$INSTALL_DEV" = "true" ]; then setup_args="$setup_args --dev"; fi && \
    bash scripts/setup_env.sh $setup_args && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# ==============================================
# Phase 2: Development Environment
# ==============================================
FROM base AS dev

COPY Makefile ./
COPY src/ ./src/
COPY train-config/ ./train-config/

CMD ["/bin/bash"]


# ==============================================
# Phase 3: Training Environment
# ==============================================
FROM base AS train

COPY Makefile ./
COPY src/ ./src/
COPY train-config/ ./train-config/

CMD ["/bin/bash"]



# ============================================
# Phase 4: Production Environment (Runtime Only)
# ============================================
FROM python:3.12-slim AS prod

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Production needs media/runtime libraries, not compilers, Python headers, uv,
# or the bootstrap utilities retained by the development/training stages.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=base /app/.venv /app/.venv
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY train-config/ ./train-config/
COPY deploy-config/ ./deploy-config/

COPY deploy/__init__.py ./deploy/__init__.py
COPY deploy/config.py ./deploy/config.py
COPY deploy/types.py ./deploy/types.py
COPY deploy/web/ ./deploy/web/
COPY deploy/inference/ ./deploy/inference/

# Expose FastAPI ports
EXPOSE 8000

# Start the ASGI application directly. This avoids executing deploy.web.api as
# __main__ and then importing it a second time through uvicorn.run().
CMD [".venv/bin/python", "-m", "uvicorn", "deploy.web.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
