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

COPY pyproject.toml uv.lock requirements.txt Makefile ./
# Keep README.md: pyproject.toml declares readme = "README.md", and Hatchling
# needs it when uv builds/installs the local project during setup.
# Keep LICENSE available in the image for package/license metadata and distribution compliance.
COPY README.md LICENSE ./

COPY scripts/ ./scripts/

ARG BOOTSTRAP_ARGS="--mirror"
ARG SETUP_ARGS="--mirror"
ARG USE_BUILD_PROXY=false

# Install system and Python dependencies
RUN if [ "$USE_BUILD_PROXY" != "true" ]; then \
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
    fi && \
    bash scripts/bootstrap.sh ${BOOTSTRAP_ARGS} && \
    bash scripts/setup_env.sh ${SETUP_ARGS} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# ==============================================
# Phase 2: Training Environment
# ==============================================
FROM base AS train

COPY src/ ./src/
COPY config/ ./config/

CMD ["/bin/bash"]



# ============================================
# Phase 3: Production Environment (Runtime Only)
# ============================================
FROM base AS prod

COPY src/config/ ./src/config/
COPY src/data/ ./src/data/
COPY src/models/ ./src/models/

COPY deploy/core/ ./deploy/core/
COPY deploy/api.py ./deploy/api.py
COPY deploy/cli.py ./deploy/cli.py

# Expose FastAPI ports
EXPOSE 8000

# Start service
CMD [".venv/bin/python", "-m", "deploy.api"]
