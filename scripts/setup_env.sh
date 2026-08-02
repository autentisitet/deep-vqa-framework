#!/usr/bin/env bash
# --- setup_env.sh ---
#
# Purpose: Set up and configure the project development environment
#
# Key tasks:
# 1. Install system dependencies (apt)
# 2. Install the uv package manager
# 3. Create a Python virtual environment (.venv)
# 4. Install core Python dependencies
# 5. Install optional tools (dev/security)
# 6. Configure PyTorch (GPU/CPU)
# 7. Verify installation
#
# Usage: ./setup_env.sh [--mirror] [--dev] [--security] [--all]


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

# ============================================================
# Logging functions
# ============================================================
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ============================================================
# Help
# ============================================================
show_help() {
    cat << EOF
Usage: ./setup_env.sh [OPTIONS]

Options:
  --mirror         Use TUNA mirror for faster downloads in China.
  --dev            Install development tools (ruff, mypy, black, isort).
  --security       Install security tools (pip-audit, cyclonedx-bom, safety).
  --all            Install everything (mirror + dev + security).
  --help, -h       Show this help message.

Examples:
  ./setup_env.sh --mirror             # Use mirror only
  ./setup_env.sh --dev                # Install dev tools
  ./setup_env.sh --security           # Install security tools
  ./setup_env.sh --all                # Install everything
EOF
    exit 0
}

# ============================================================
# Parse arguments
# ============================================================
USE_MIRROR=false
INSTALL_DEV=false
INSTALL_SECURITY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)   USE_MIRROR=true; shift ;;
        --dev)      INSTALL_DEV=true; shift ;;
        --security) INSTALL_SECURITY=true; shift ;;
        --all)      USE_MIRROR=true; INSTALL_DEV=true; INSTALL_SECURITY=true; shift ;;
        --help|-h)  show_help ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# ============================================================
# Sudo detection (kept original logic)
# ============================================================
HAS_SUDO=false
command -v sudo &> /dev/null && HAS_SUDO=true

if [ "$(id -u)" -eq 0 ]; then
    export APP_SUDO=""
    export ADMIN_SUDO=""
elif [ "$OSTYPE" = darwin* ]; then
    export APP_SUDO=""
    if [ "$HAS_SUDO" = true ]; then
        export ADMIN_SUDO="sudo"
    else
        export ADMIN_SUDO=""
    fi
else
    if [ "$HAS_SUDO" = true ]; then
        export APP_SUDO="sudo"
        export ADMIN_SUDO="sudo"
    else
        log_warn "Not root and 'sudo' not found. Trying without it..."
        export APP_SUDO=""
        export ADMIN_SUDO=""
    fi
fi

# ============================================================
# Project directories
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"
DETECTED_PATH="${PROJECT_DIR}/datasets"

export UV_CACHE_DIR="$PROJECT_PARENT_DIR/.uv_cache"
PYTHON_VERSION="3.12"

mkdir -p "$PROJECT_DIR" "$DETECTED_PATH"
cd "$PROJECT_DIR"

# ============================================================
# Package lists
# ============================================================
APT_PACKAGES=(
    curl wget build-essential tar aria2 unrar unzip tree bc ffmpeg
)
UV_CORE_PACKAGES=(
    opencv-python decord pyyaml
    numpy pandas matplotlib pillow seaborn
    loguru tqdm
    scikit-learn scipy
    gdown
)
UV_DEV_PACKAGES=(ruff mypy black isort)
UV_SECURITY_PACKAGES=(pip-audit cyclonedx-bom safety)

# ============================================================
# Proxy check
# ============================================================
check_proxy() {
    if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
        log_info "Using proxy from environment: ${http_proxy:-$HTTP_PROXY}"
        return 0
    else
        log_info "No proxy set. If needed:"
        echo "   - On AutoDL: source /etc/network_turbo"
        echo "   - Other: set http_proxy/https_proxy environment variables"
        return 1
    fi
}

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
# Download helper with fallback
# ============================================================
download_with_fallback() {
    local url="$1"
    local output="$2"

    if command -v aria2c &> /dev/null; then
        aria2c -x 4 -s 4 -o "$output" --timeout=30 --max-tries=3 --console-log-level=error "$url" 2>/dev/null && return 0
    fi
    if command -v wget &> /dev/null; then
        wget -q -O "$output" --timeout=30 --tries=3 "$url" 2>/dev/null && return 0
    fi
    if command -v curl &> /dev/null; then
        curl -L -o "$output" --connect-timeout 30 --retry 3 "$url" 2>/dev/null && return 0
    fi
    return 1
}

# ============================================================
# Main execution
# ============================================================
log_info "Installing basic tools..."
check_proxy

# Configure mirror if requested
if [ "$USE_MIRROR" = true ]; then
    log_info "Using TUNA mirror for apt and pip..."
    TEMP_SOURCES="/tmp/tuna_sources.list"
    CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
    cat > "$TEMP_SOURCES" <<EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $CODENAME main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $CODENAME-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $CODENAME-security main restricted universe multiverse
EOF
    APT_OPT=(
        "-o" "Dir::Etc::SourceList=$TEMP_SOURCES"
        "-o" "Dir::Etc::SourceParts=/dev/null"
    )
    export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

# Install system dependencies
${APP_SUDO} apt-get update ${APT_OPT[@]} -qq
${APP_SUDO} apt-get install -y ${APT_OPT[@]} "${APT_PACKAGES[@]}"
log_ok "System dependencies installed."

# ============================================================
# Install uv
# ============================================================
if ! command -v uv &> /dev/null; then
    log_info "Installing uv..."
    if download_with_fallback "https://astral.sh/uv/install.sh" "/tmp/uv_install.sh"; then
        bash /tmp/uv_install.sh && rm -f /tmp/uv_install.sh
        [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
        log_ok "uv installed successfully."
    else
        log_error "All download methods failed. Please check network."
        exit 1
    fi
else
    log_ok "uv already installed: $(uv --version)"
fi

export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.cargo/bin:$PATH"

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
log_info "Checking dependencies..."

MISSING_PACKAGES=()
for pkg in "${UV_CORE_PACKAGES[@]}"; do
    if ! uv pip show "$pkg" &> /dev/null 2>&1; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log_info "Adding missing packages: ${MISSING_PACKAGES[*]}"
    uv add "${MISSING_PACKAGES[@]}" --no-sync 2>/dev/null || log_warn "Failed to add packages individually, trying full sync..."

    SYNC_ARGS=""
    [ -n "$UV_INDEX_URL" ] && SYNC_ARGS="$SYNC_ARGS --index-url $UV_INDEX_URL"
    [ "$USE_TORCH_INDEX" = true ] && [ -n "$TORCH_INDEX" ] && SYNC_ARGS="$SYNC_ARGS --extra-index-url $TORCH_INDEX"
    SYNC_ARGS="$SYNC_ARGS --no-dev"
    uv sync $SYNC_ARGS
else
    log_ok "All core packages already present. Skipping sync."
fi

# ============================================================
# Install optional tools
# ============================================================
if [ "$INSTALL_DEV" = true ]; then
    log_info "Installing development tools..."
    ensure_optional_deps
    uv add --optional dev "${UV_DEV_PACKAGES[@]}" 2>/dev/null || true
    log_ok "Development tools installed:"
    echo "  • ruff (code linting & formatting)"
    echo "  • mypy (type checking)"
    echo "  • black (code formatter)"
    echo "  • isort (import sorting)"
else
    log_info "Skipping development tools (use --dev to install)."
fi

if [ "$INSTALL_SECURITY" = true ]; then
    log_info "Installing security tools..."
    ensure_optional_deps
    uv add --optional security "${UV_SECURITY_PACKAGES[@]}" 2>/dev/null || true
    log_ok "Security tools installed:"
    echo "  • pip-audit (vulnerability scanning)"
    echo "  • cyclonedx-bom (SBOM generation)"
    echo "  • safety (dependency security check)"
else
    log_info "Skipping security tools (use --security to install)."
fi

# Clean up temporary mirror config
[ -n "$TEMP_SOURCES" ] && [ -f "$TEMP_SOURCES" ] && rm -f "$TEMP_SOURCES"

# ============================================================
# Verify installation
# ============================================================
echo ""
log_info "Verifying installation..."
echo "----------------------------------------------------------------------"

uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || log_warn "PyTorch verification failed"

uv run python -c "import cv2, numpy, pandas, loguru, tqdm, sklearn, scipy, gdown; print('All core dependencies imported successfully')" 2>/dev/null || log_warn "Some dependencies failed to import"

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