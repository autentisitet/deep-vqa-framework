#!/usr/bin/env bash
# --- bootstrap.sh ---
# System-level initialization: apt, mirrors, system tools

set -e -o pipefail

# ============================================================
# Color definitions
# ============================================================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }



# ============================================================
# Argument parsing
# ============================================================
USE_MIRROR=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror) USE_MIRROR=true; shift ;;
        --help|-h)
            cat << EOF
Usage: ./bootstrap.sh [--mirror]

Options:
  --mirror   Use TUNA mirror for faster downloads in China
  --help     Show this help message
EOF
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

APT_OPT=()


# ============================================================
# Check permissions
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
# Package list
# ============================================================
APT_PACKAGES=(
    make curl wget build-essential tar aria2 unrar unzip tree bc ffmpeg jq
)

if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    log_info "pip not found, adding python3-pip to install list..."
    APT_PACKAGES+=(python3-pip)
fi

if [ ! -f /usr/include/python3*/Python.h ] 2>/dev/null; then
    log_info "Python dev headers not found, adding python3-dev..."
    APT_PACKAGES+=(python3-dev)
fi


# ============================================================
# Configure mirror sources
# ============================================================
if [ "$USE_MIRROR" = true ]; then
    log_info "Using TUNA mirror for apt..."
    TEMP_SOURCES="/tmp/tuna_sources.list"
    
    # 检测系统类型和代号
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        CODENAME=${VERSION_CODENAME:-$UBUNTU_CODENAME}
        # 检测是 Debian 还是 Ubuntu
        if [ "$ID" = "debian" ]; then
            MIRROR_PATH="debian"
            COMPONENTS="main contrib non-free non-free-firmware"
        elif [ "$ID" = "ubuntu" ]; then
            MIRROR_PATH="ubuntu"
            COMPONENTS="main restricted universe multiverse"
        else
            log_error "Unsupported distribution: $ID"
            exit 1
        fi
    else
        log_error "Cannot detect OS version"
        exit 1
    fi
    
    cat > "$TEMP_SOURCES" <<EOF
deb https://mirrors.tuna.tsinghua.edu.cn/$MIRROR_PATH/ $CODENAME $COMPONENTS
deb https://mirrors.tuna.tsinghua.edu.cn/$MIRROR_PATH/ $CODENAME-updates $COMPONENTS
EOF
    
    # Debian 和 Ubuntu 的安全源路径不同
    if [ "$ID" = "debian" ]; then
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security/ $CODENAME-security $COMPONENTS" >> "$TEMP_SOURCES"
    else
        echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $CODENAME-security $COMPONENTS" >> "$TEMP_SOURCES"
    fi
    
    APT_OPT=(
        "-o" "Dir::Etc::SourceList=$TEMP_SOURCES"
        "-o" "Dir::Etc::SourceParts=/dev/null"
    )
fi



# ============================================================
# Install system dependencies
# ============================================================
log_info "Installing system dependencies..."

${APP_SUDO} apt-get update ${APT_OPT[@]} -qq
${APP_SUDO} apt-get install -y ${APT_OPT[@]} "${APT_PACKAGES[@]}"


[ -n "$TEMP_SOURCES" ] && [ -f "$TEMP_SOURCES" ] && rm -f "$TEMP_SOURCES"
log_ok "System dependencies installed."
