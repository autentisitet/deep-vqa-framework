#!/usr/bin/env bash
# --- setup_env.sh ---


# Tasks:
# 1. switch the correct file path
# 2. Download datasets unless AutoDL isn't exist
# 3. verify the dataset hash
# 4. unzip the dataset
# 5. install uv, apt modules, python modules...etc
# 6. set the environment configs
# 7. write the base_config.yaml
set -e -o pipefail

show_help() {
    echo "Usage: ./setup_env.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mirror         Use TUNA mirror for faster downloads in China."
    echo "  --dev            Install development tools (ruff, mypy, black, isort)."
    echo "  --security       Install security tools (pip-audit, cyclonedx-bom, safety)."
    echo "  --all            Install everything (mirror + dev + security)."
    echo "  --h, --help      Show this help message."
    echo ""
    echo "Examples:"
    echo "  ./setup_env.sh --mirror             # Use mirror only"
    echo "  ./setup_env.sh --dev                # Install dev tools"
    echo "  ./setup_env.sh --security           # Install security tools"
    echo "  ./setup_env.sh --all                # Install everything"
    echo ""
    exit 0
}


USE_MIRROR=false
INSTALL_DEV=false
INSTALL_SECURITY=false


while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)
            USE_MIRROR=true
            shift ;;
        --dev)
            INSTALL_DEV=true
            shift ;;
        --security)
            INSTALL_SECURITY=true
            shift ;;
        --all)
            USE_MIRROR=true
            INSTALL_DEV=true
            INSTALL_SECURITY=true
            shift ;;
        --help|-h)
            show_help ;;
        *)
            echo -e "\033[1;31mUnknown option: $1\033[0m"
            show_help ;;
    esac
done



HAS_SUDO=false
if command -v sudo &> /dev/null; then
    HAS_SUDO=true
fi


if [ "$(id -u)" -eq 0 ]; then
    export APP_SUDO=""
    export ADMIN_SUDO=""
elif [ "$OSTYPE" == darwin* ]; then
    # macOS does not require sudo except for writing to /Library and modifying system configuration.
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
        echo -e "\033[1;33mWarning: Not root and 'sudo' not found. Trying without it...\033[0m"
        export APP_SUDO=""
        export ADMIN_SUDO=""
    fi
fi




detect_proxy_port() {
    if [ -n "$http_proxy" ]; then
        local port=$(echo "$http_proxy" | sed -E 's/.*:([0-9]+).*/\1/')
        if curl -s -o /dev/null --max-time 2 --proxy "$http_proxy" "https://httpbin.org/get" 2>/dev/null; then
            echo "$port"
            return 0
        fi
    fi


    for port in 7890 7897 10809 1080; do
        if curl -s -o /dev/null --max-time 2 --proxy "http://127.0.0.1:$port" "https://httpbin.org/get" 2>/dev/null; then
            echo "$port"
            return 0
        fi
    done

    return 1
}


setup_proxy() {
    local port
    port=$(detect_proxy_port) || true
    if [ -n "$port" ]; then
        export http_proxy="http://127.0.0.1:$port"
        export https_proxy="$http_proxy"
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.cn,*.mirrors.edu.cn,mirrors.tuna.tsinghua.edu.cn"
        echo "Proxy enabled on port $port"
    else
        echo "No proxy detected, using direct connection"
    fi
}


ensure_optional_deps() {
    if ! grep -q "\[project.optional-dependencies\]" pyproject.toml 2>/dev/null; then
        echo "" >> pyproject.toml
        echo "[project.optional-dependencies]" >> pyproject.toml
        echo "dev = []" >> pyproject.toml
        echo "security = []" >> pyproject.toml
    fi
}



# Output styling
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'




# ============================================================
# Configure project directories
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"
DETECTED_PATH="${PROJECT_DIR}/datasets"



# ============================================================
# Configure uv cache directory, python version, and ensure the cache directory exists
# ============================================================
export UV_CACHE_DIR="$PROJECT_PARENT_DIR/.uv_cache"
PYTHON_VERSION="3.12"
export UV_CACHE_DIR

mkdir -p "$PROJECT_DIR"
mkdir -p "$DETECTED_PATH"
cd "$PROJECT_DIR"



# ============================================================
# Configure package lists
# ============================================================
APT_PACKAGES=(
    curl
    wget
    net-tools
    iproute2
    psmisc
    build-essential
    tar
    aria2
    unrar
    unzip
    tree
    bc
    ffmpeg
    imagemagick
    dos2unix
)
UV_CORE_PACKAGES=(
    opencv-python decord
    pyyaml
    numpy pandas matplotlib pillow seaborn
    loguru tqdm rich PyYAML
    scikit-learn scipy
    gdown
)
UV_DEV_PACKAGES=(
    ruff mypy black isort
)
UV_SECURITY_PACKAGES=(
    pip-audit cyclonedx-bom safety
)


# ============================================================
# Install basic tools and configure mirrors if needed
# ===========================================================
echo "⚙️ Installing basic tools..."
if [ -f "/etc/network_environment" ]; then
    source /etc/network_environment
else
    setup_proxy
fi


if [ "$USE_MIRROR" = true ]; then
    echo -e "${GREEN}Using temporary TUNA mirror config...${NC}"
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

${APP_SUDO} apt-get update ${APT_OPT[@]} -qq
${APP_SUDO} apt-get install -y ${APT_OPT[@]} "${APT_PACKAGES[@]}"




# ============================================================
# Install uv if not present
# ============================================================
if ! command -v uv &> /dev/null; then
    UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    INSTALLED=false

    if command -v aria2c &> /dev/null; then
        echo "📥 Using aria2 for faster download..."
        if aria2c -x 4 -s 4 \
               -o /tmp/uv_install.sh \
               --user-agent="$UA" \
               --timeout=30 \
               --max-tries=3 \
               --console-log-level=error \
               --allow-overwrite=true \
               https://astral.sh/uv/install.sh; then
            bash /tmp/uv_install.sh && rm -f /tmp/uv_install.sh && INSTALLED=true
        else
            echo "⚠️ aria2 failed, trying next method..."
        fi
    fi


    if [ "$INSTALLED" = false ] && command -v wget &> /dev/null; then
        echo "📥 Using wget..."
        if wget -q --show-progress \
             -O /tmp/uv_install.sh \
             -c \
             -U "$UA" \
             --timeout=30 \
             --tries=3 \
             https://astral.sh/uv/install.sh; then
            bash /tmp/uv_install.sh && rm -f /tmp/uv_install.sh && INSTALLED=true
        else
            echo "⚠️ wget failed, trying next method..."
        fi
    fi


    if [ "$INSTALLED" = false ] && command -v curl &> /dev/null; then
        echo "📥 Using curl..."
        if curl -# -L \
             -A "$UA" \
             --connect-timeout 30 \
             --retry 3 \
             --output /tmp/uv_install.sh \
             https://astral.sh/uv/install.sh; then
            bash /tmp/uv_install.sh && rm -f /tmp/uv_install.sh && INSTALLED=true
        else
            echo "⚠️ curl failed"
        fi
    fi


    if [ "$INSTALLED" = true ]; then
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
        echo -e "${GREEN}✅ uv installed${NC}"
    else
        echo -e "${RED}❌ All download methods failed${NC}"
        echo "   Please check your network connection and try again."
        exit 1
    fi
else
    echo -e "${GREEN}✅ uv already installed: $(uv --version)${NC}"
fi

export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.cargo/bin:$PATH"


# ============================================================
# Check if pyproject.toml exists and initialize uv if not
# ============================================================
if [ -f "pyproject.toml" ]; then
    echo -e "${GREEN}✅ pyproject.toml found. Skipping uv init.${NC}"
else
    echo -e "${BLUE}⚙️  No pyproject.toml found. Initializing...${NC}"
    uv init --bare
fi

# Ensure the project name is correct.
sed -i 's/name = ".*"/name = "deep-vqa-framework"/' pyproject.toml


# ============================================================
# Detect GPU for PyTorch and set the appropriate index URL for torch installation
# ============================================================
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    if ! [[ "$CUDA_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "⚠ Could not reliably detect CUDA version, falling back to 12.1"
        CUDA_VERSION="12.1"
    fi

    CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | tr -d '.')
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_MAJOR_MINOR}"
    echo "✔ Detected GPU: Using CUDA $CUDA_VERSION"
    USE_TORCH_INDEX=true
else
    echo "⚠ Detected CPU only: Configuring standard torch."
    TORCH_INDEX=""
    USE_TORCH_INDEX=false
fi




# ============================================================
# Setup Python virtual environment and install dependencies
# ============================================================
echo "⚙️ Setting up Python environment..."

if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    echo -e "${GREEN}✅ .venv already exists.${NC}"
else
    echo -e "${BLUE}📂 Creating new .venv with Python ${PYTHON_VERSION}...${NC}"
    uv venv .venv --python "$PYTHON_VERSION" --seed
fi


# Pin the Python version
echo "📌 Pinning Python version to ${PYTHON_VERSION}..."
uv python pin "$PYTHON_VERSION" 2>/dev/null || true


# Check and install missing packages
echo "Checking dependencies..."
MISSING_PACKAGES=()
for pkg in "${UV_CORE_PACKAGES[@]}"; do
    if ! uv pip show "$pkg" &> /dev/null 2>&1; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${BLUE}📦 Adding missing packages: ${MISSING_PACKAGES[*]}${NC}"
    if ! uv add "${MISSING_PACKAGES[@]}" --no-sync 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Failed to add packages individually, trying full sync...${NC}"
    fi
    echo "Syncing dependencies..."
    # 构建 sync 命令参数
    SYNC_ARGS=""
    if [ -n "$UV_INDEX_URL" ]; then
        SYNC_ARGS="$SYNC_ARGS --index-url $UV_INDEX_URL"
    fi
    if [ "$USE_TORCH_INDEX" = true ] && [ -n "$TORCH_INDEX" ]; then
        SYNC_ARGS="$SYNC_ARGS --extra-index-url $TORCH_INDEX"
    fi
    SYNC_ARGS="$SYNC_ARGS --no-dev"
    # 直接执行
    uv sync $SYNC_ARGS
else
    echo -e "${GREEN}✅ All packages already present. Skipping sync.${NC}"
fi


echo -e "${BLUE}🔧 Installing security tools...${NC}"

# Check and create optional-dependencies configuration
# Install development tools if requested
if [ "$INSTALL_DEV" = true ]; then
    echo -e "${BLUE}🔧 Installing development tools...${NC}"
    ensure_optional_deps
    
    uv add --optional dev "${UV_DEV_PACKAGES[@]}" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Development tools installed:${NC}"
    echo "  • ruff (code linting & formatting)"
    echo "  • mypy (type checking)"
    echo "  • black (code formatter)"
    echo "  • isort (import sorting)"
else
    echo -e "${YELLOW}ℹ️  Skipping development tools.${NC}"
    echo "   Use --dev to install them."
fi

# Install security tools if requested
if [ "$INSTALL_SECURITY" = true ]; then
    echo -e "${BLUE}🔒 Installing security tools...${NC}"
    ensure_optional_deps
    
    uv add --optional security "${UV_SECURITY_PACKAGES[@]}" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Security tools installed:${NC}"
    echo "  • pip-audit (vulnerability scanning)"
    echo "  • cyclonedx-bom (SBOM generation)"
    echo "  • safety (dependency security check)"
else
    echo -e "${YELLOW}ℹ️  Skipping security tools.${NC}"
    echo "   Use --security to install them."
fi


# Non-intrusive configuration apt mirror source
[ -n "$TEMP_SOURCES" ] && [ -f "$TEMP_SOURCES" ] && rm -f "$TEMP_SOURCES"



# ============================================================
# Verify installation and configuration
# ============================================================
echo "------------------------------------------------"
echo "🔍 Verifying installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import cv2, numpy, pandas, loguru, tqdm, sklearn, scipy, gdown; print('✅ All dependencies imported successfully')"
echo ""
echo "------------------------------------------------"
echo "🚀 Configuration complete!"
echo "   Python Version: ${PYTHON_VERSION}"
echo "   Virtual Env: $(pwd)/.venv"
echo "   Dev Tools: $([ "$INSTALL_DEV" = true ] && echo "${GREEN}Installed.${NC}" || echo "${YELLOW}Skipped.${NC}")"
echo "   Security Tools: $([ "$INSTALL_SECURITY" = true ] && echo "${GREEN}Installed.${NC}" || echo "${YELLOW}Skipped.${NC}")"
echo ""
if [ "$INSTALL_DEV" = false ]; then
    echo "💡 To install dev tools later:"
    echo "   uv add --optional dev ruff mypy black isort"
fi
if [ "$INSTALL_SECURITY" = false ]; then
    echo "💡 To install security tools later:"
    echo "   uv add --optional security pip-audit cyclonedx-bom safety"
fi
echo ""
echo "🔔 IMPORTANT: uv is installed at ~/.local/bin/uv"
echo ""
echo "   To use uv in your current shell, run:"
echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "   To make this permanent, add to ~/.bashrc:"
echo "   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
echo "------------------------------------------------"