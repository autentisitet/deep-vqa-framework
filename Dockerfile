# Dockerfile - 多阶段构建

# ============================================
# 阶段1: 基础环境（共享层）
# ============================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY README.md LICENSE ./
COPY pyproject.toml uv.lock requirements.txt Makefile ./
COPY scripts/ ./scripts/

# 安装系统依赖、Python 依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends make && \
    make bootstrap BOOTSTRAP_ARGS="--mirror" && \
    make setup SETUP_ARGS="--mirror" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# ============================================
# 阶段2: 训练环境
# ============================================
FROM base AS train

COPY src/ ./src/
COPY config/ ./config/

CMD ["/bin/bash"]



# ============================================
# 阶段3: 生产环境（只含运行所需）
# ============================================
FROM base AS prod

ENV PATH="/root/.local/bin:$PATH"

COPY deploy/ ./deploy/
COPY frontend/ ./frontend/

# 清理不需要的文件（减小镜像体积）
RUN find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -type d -exec rm -rf {} + \
    && rm -rf /app/scripts /app/tests /app/docs 2>/dev/null || true


# 暴露 FastAPI 端口
EXPOSE 8000

# 启动服务
CMD ["uv", "run", "python", "-m", "deploy.api"]