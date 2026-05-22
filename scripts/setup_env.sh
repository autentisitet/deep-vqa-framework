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
set -e

show_help() {
    echo "Usage: ./setup_env.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mirror     Use TUNA (Tsinghua University) mirror for faster downloads in China."
    echo "  --h, --help      Show this help message."
    echo ""
    exit 0
}


USE_MIRROR=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)
            USE_MIRROR=true
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


# Output styling
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'


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
    ffprobe
)
UV_PACKAGES=(
    torch torchvision torchaudio
    opencv-python decord
    pyyaml
    numpy matplotlib pillow seaborn
    loguru tqdm rich PyYAML
    scikit-learn scipy
    gdown
)
DATA_DISK_PATH="/root/autodl-tmp"
PROJECT_DIR="$DATA_DISK_PATH/deep-vqa-framework"
DETECTED_PATH="${PROJECT_DIR}/datasets"

rm -f "$DATA_DISK_PATH/pyproject.toml" "$DATA_DISK_PATH/uv.lock" "$DATA_DISK_PATH/.python-version"
rm -rf "$DATA_DISK_PATH/.venv"
mkdir -p "$PROJECT_DIR"
mkdir -p "$DETECTED_PATH"


cd "$PROJECT_DIR"
echo "⚙️ Installing basic tools..."
if [ -f "/etc/network_environment" ]; then
    source /etc/network_environment
# else
  # export http_proxy=http://127.0.0.1:7890
  # export https_proxy=http://127.0.0.1:7890
  # export no_proxy="127.0.0.1,localhost,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.cn,*.mirrors.edu.cn,mirrors.tuna.tsinghua.edu.cn"
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

${APT_SUDO} apt-get update ${APT_OPT[@]} -qq
${APT_SUDO} apt-get install -y ${APT_OPT[@]} "${APT_PACKAGES[@]}"



export UV_CACHE_DIR="$DATA_DISK_PATH/.uv_cache"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"
fi


# cd "$(dirname "$0")/.."
[ ! -f "pyproject.toml" ] && uv init --bare     # Initialize the bare pyproject.toml
sed -i 's/name = ".*"/name = "deep-vqa-framework"/' pyproject.toml

# There may be path conflicts in the Conda environment,
#   or incompatibility between dependencies in the old environment (Resolution failure),
#   which can cause the installation of the UV environment to be very slow.
PYTHON_EXE=$(which python3 || which python)
if [ -z "$PYTHON_EXE" ]; then
    echo "[!] System Python not found, letting uv handle it..."
    uv venv --python 3.12 --seed --clear
else
    echo "[√] Found Python at $PYTHON_EXE"
    uv venv --python "$PYTHON_EXE" --clear
fi
source .venv/bin/activate
echo "📌 Pinning Python version and syncing packages..."
uv python pin 3.12      # Lock the python version


uv add "${UV_PACKAGES[@]}" --no-sync
uv sync --project ./pyproject.toml

# Non-intrusive configuration apt mirror source
[ -n "$TEMP_SOURCES" ] && [ -f "$TEMP_SOURCES" ] && rm -f "$TEMP_SOURCES"



echo "------------------------------------------------"
echo "🚀 Configuration complete! Execute the command to start training:"
echo "   chmod +x ./manage_data.sh      "
echo "   ./manage_data.sh               "
echo "   nohup uv run main.py > train.log 2>&1"
echo "------------------------------------------------"