#!/usr/bin/env bash
# --- setup_env.sh ---
# Project-level initialization: uv, .venv, Python dependencies
# Does NOT touch system packages (apt, sudo, mirrors)
# Works on both host machine and container


set -e -o pipefail


# ============================================================
# Color definitions
# ============================================================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'


log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }



# ============================================================
# Parse arguments
# ============================================================
INSTALL_DEV=false
INSTALL_SECURITY=false
USE_MIRROR=false


while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)      INSTALL_DEV=true; shift ;;
        --security) INSTALL_SECURITY=true; shift ;;
        --all)      INSTALL_DEV=true; INSTALL_SECURITY=true; shift ;;
        --mirror)   USE_MIRROR=true; shift ;;
        --help|-h)
            cat << EOF
Usage: ./setup_env.sh [OPTIONS]

Options:
  --dev         Install development tools (pytest, ruff, mypy, black, isort)
  --security    Install security tools (pip-audit, cyclonedx-bom, safety)
  --all         Install all optional tools
  --mirror      Use TUNA mirror for pip (China users)
  --help        Show this help message
EOF
            exit 0
            ;;
        *) log_warn "Unknown option: $1"; shift ;;
    esac
done



# ============================================================
# Environment setup
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DETECTED_PATH="${PROJECT_DIR}/datasets"
mkdir -p "$DETECTED_PATH"

if [ -z "$UV_CACHE_DIR" ]; then
    export UV_CACHE_DIR="$HOME/.cache/uv"
fi

PYTHON_VERSION="3.12"


if [ "$USE_MIRROR" = true ]; then
    if [ -z "$UV_INDEX_URL" ]; then
        export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
    if [ -z "$PIP_INDEX_URL" ]; then
        export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
    log_info "Using TUNA mirror for pip"
fi




# ============================================================
# Package lists
# ============================================================
UV_DEV_PACKAGES=(pytest ruff mypy black isort)
UV_SECURITY_PACKAGES=(pip-audit cyclonedx-bom safety)



# ============================================================
# Ensure optional-dependencies section in pyproject.toml
# ============================================================
ensure_optional_deps() {
    if ! grep -q "\[project.optional-dependencies\]" pyproject.toml 2>/dev/null; then
        echo "" >> pyproject.toml
        echo "[project.optional-dependencies]" >> pyproject.toml
        echo "dev = []" >> pyproject.toml
        echo "security = []" >> pyproject.toml
    fi
}




# ============================================================
# Setup proxy (for pip install)
# ============================================================
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.cargo/bin:$PATH"


_proxy_http="${http_proxy:-$HTTP_PROXY}"
_proxy_https="${https_proxy:-$HTTPS_PROXY}"


if [ -n "$_proxy_http" ] || [ -n "$_proxy_https" ]; then
    export http_proxy="$_proxy_http"
    export https_proxy="$_proxy_https"
    export HTTP_PROXY="$_proxy_http"
    export HTTPS_PROXY="$_proxy_https"
    log_info "Using proxy: ${_proxy_http:-${_proxy_https}}"
else
    log_info "No proxy set. Proceeding with direct connection."
    echo "   (Tip: Run 'source /etc/network_turbo' first if downloading on AutoDL)"
fi




# ============================================================
# Install uv via pip (works on both host and container)
# ============================================================
if ! command -v uv &> /dev/null; then
    log_info "Installing uv via pip..."

    if ! command -v pip &> /dev/null; then
        log_error "pip not found. Please ensure Python and pip are installed."
        exit 1
    fi

    if [ -n "$PIP_INDEX_URL" ]; then
        PIP_INSTALL_ARGS="pip install uv -i $PIP_INDEX_URL"
    elif [ -n "$UV_INDEX_URL" ]; then
        PIP_INSTALL_ARGS="pip install uv -i $UV_INDEX_URL"
    else
        PIP_INSTALL_ARGS="pip install uv"
    fi


    if [ "$(id -u)" -eq 0 ]; then
        $PIP_INSTALL_ARGS 2>/dev/null || {
            log_error "pip install uv failed"
            exit 1
        }
    else
        $PIP_INSTALL_ARGS --user 2>/dev/null || {
            log_error "pip install uv --user failed"
            exit 1
        }
    fi

    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        log_ok "uv installed successfully: $(uv --version)"
    else
        log_error "uv installation failed: command not found after install"
        exit 1
    fi
else
    log_ok "uv already installed: $(uv --version)"
fi


# ============================================================
# Initialize pyproject.toml
# ============================================================
if [ -f "pyproject.toml" ]; then
    log_ok "pyproject.toml found."
else
    log_info "No pyproject.toml found. Initializing..."
    uv init --bare
fi

sed -i 's/name = ".*"/name = "deep-vqa-framework"/' pyproject.toml



# ============================================================
# Detect GPU and configure PyTorch
# ============================================================
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    if ! [[ "$CUDA_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
        log_warn "Could not reliably detect CUDA version, falling back to 12.1"
        CUDA_VERSION="12.1"
    fi
    CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | tr -d '.')
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_MAJOR_MINOR}"
    log_info "Detected GPU: Using CUDA $CUDA_VERSION"
    USE_TORCH_INDEX=true
else
    log_info "No GPU detected. Configuring CPU-only torch."
    TORCH_INDEX=""
    USE_TORCH_INDEX=false
fi



# ============================================================
# Setup Python virtual environment
# ============================================================
log_info "Setting up Python environment..."

if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    log_ok ".venv already exists."
else
    log_info "Creating new .venv with Python ${PYTHON_VERSION}..."
    uv venv .venv --python "$PYTHON_VERSION" --seed
fi

uv python pin "$PYTHON_VERSION" 2>/dev/null || true



# ============================================================
# Install Python dependencies
# ============================================================
log_info "Installing Python dependencies via uv sync..."

SYNC_ARGS=""

if [ "$USE_TORCH_INDEX" = true ] && [ -n "$TORCH_INDEX" ]; then
    SYNC_ARGS="$SYNC_ARGS --extra-index-url $TORCH_INDEX"
fi

if [ -n "$UV_INDEX_URL" ]; then
    SYNC_ARGS="$SYNC_ARGS --index-url $UV_INDEX_URL"
fi

uv sync --no-dev $SYNC_ARGS
log_ok "Core dependencies installed."



# ============================================================
# Install build-system dependencies (for uv run)
# ============================================================
log_info "Installing build-system dependencies (hatchling)..."
uv pip install hatchling 2>/dev/null || log_warn "hatchling installation failed"
log_ok "Build-system dependencies installed."



# ============================================================
# Install optional tools
# ============================================================
if [ "$INSTALL_DEV" = true ] || [ "$INSTALL_SECURITY" = true ]; then
    ensure_optional_deps
fi

if [ "$INSTALL_DEV" = true ]; then
    log_info "Installing development tools..."
    uv add --optional dev "${UV_DEV_PACKAGES[@]}" 2>/dev/null || true
    if ! uv run pytest --version >/dev/null 2>&1; then
        log_error "pytest installation failed. Run 'make setup SETUP_ARGS=\"--dev\"' again."
        exit 1
    fi
    log_ok "Development tools installed:"
    echo "  • ruff (code linting & formatting)"
    echo "  • pytest (test runner, verified)"
    echo "  • mypy (type checking)"
    echo "  • black (code formatter)"
    echo "  • isort (import sorting)"
fi

if [ "$INSTALL_SECURITY" = true ]; then
    log_info "Installing security tools..."
    uv add --optional security "${UV_SECURITY_PACKAGES[@]}" 2>/dev/null || true
    log_ok "Security tools installed:"
    echo "  • pip-audit (vulnerability scanning)"
    echo "  • cyclonedx-bom (SBOM generation)"
    echo "  • safety (dependency security check)"
fi


# ============================================================
# Verify installation
# ============================================================
echo ""
log_info "Verifying installation..."
echo "----------------------------------------------------------------------"

uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || log_warn "PyTorch verification failed"

uv run python -c "import cv2, numpy, pandas, loguru, tqdm, sklearn, scipy, gdown; print('All core dependencies imported successfully')" 2>/dev/null || log_warn "Some dependencies failed to import"

uv run python -c "import hatchling; print('hatchling ok')" 2>/dev/null || log_warn "hatchling verification failed"
echo "----------------------------------------------------------------------"
log_ok "Setup complete!"
echo ""
echo "  Python Version: ${PYTHON_VERSION}"
echo "  Virtual Env:    $(pwd)/.venv"
echo "  Dev Tools:      $([ "$INSTALL_DEV" = true ] && echo "${GREEN}Installed${NC}" || echo "${YELLOW}Skipped${NC}")"
echo "  Security Tools: $([ "$INSTALL_SECURITY" = true ] && echo "${GREEN}Installed${NC}" || echo "${YELLOW}Skipped${NC}")"
echo ""
echo "To use uv in current shell:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "Add to ~/.bashrc:"
echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
