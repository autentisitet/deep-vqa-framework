#!/usr/bin/env bash
# --- cache_clean.sh ---
#
# Purpose: Clean and optimize disk space for Deep-VQA-Framework
#
# Key tasks:
# 1. Migrate AI model caches (HuggingFace/ModelScope) to data disk
# 2. Remove dataset archive files after extraction
# 3. Clean Python package manager caches (uv/pip)
# 4. Clean Conda and APT caches
# 5. Remove Python bytecode and Jupyter checkpoints
# 6. Show disk usage report
#
# Usage: ./cache_clean.sh

set -e -o pipefail

# ============================================================
# Color definitions (consistent with other scripts)
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
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_PARENT_DIR="$(dirname "$PROJECT_DIR")"
DOWNLOAD_CACHE="${PROJECT_DIR}/.download_cache"

# Redirect AI model and package manager caches to data disk
TMP_CACHE_DIR="${PROJECT_PARENT_DIR}/.system_caches"
mkdir -p "$TMP_CACHE_DIR/huggingface" "$TMP_CACHE_DIR/modelscope" "$TMP_CACHE_DIR/uv"

# ============================================================
# Sudo detection
# ============================================================
APP_SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    APP_SUDO="sudo"
fi

# ============================================================
# Main
# ============================================================
log_info "Starting cache cleaning and disk optimization..."

# ============================================================
# 1. AI model cache migration (symlink redirect)
# ============================================================
log_info "Optimizing AI model cache routing (HuggingFace/ModelScope)..."

for hub_name in "huggingface" "modelscope"; do
    TARGET_LINK="$HOME/.cache/${hub_name}"
    DATA_DISK_CACHE="${TMP_CACHE_DIR}/${hub_name}"

    if [ -d "$TARGET_LINK" ] && [ ! -L "$TARGET_LINK" ]; then
        log_info "Migrating ${hub_name} weights from system disk to data disk..."
        mkdir -p "$DATA_DISK_CACHE"
        cp -r "$TARGET_LINK"/* "$DATA_DISK_CACHE/" 2>/dev/null || true
        rm -rf "$TARGET_LINK"
    fi

    if [ ! -L "$TARGET_LINK" ]; then
        mkdir -p "$HOME/.cache"
        ln -snf "$DATA_DISK_CACHE" "$TARGET_LINK"
        log_ok "${hub_name} cache redirected to data disk."
    fi
done

# ============================================================
# 2. Dataset compression package removal
# ============================================================
if [ -d "$DOWNLOAD_CACHE" ]; then
    if [ -d "$PROJECT_DIR/datasets/KoNViD-1k" ] || [ -d "$PROJECT_DIR/datasets/TID2013" ]; then
        log_info "Dataset extraction verified, removing original archives..."
        rm -rf "$DOWNLOAD_CACHE"
        log_ok "Download cache removed."
    else
        log_warn "Dataset extraction not verified. Keeping original archives."
    fi
fi

# Remove zero-byte zombie files
find "$PROJECT_PARENT_DIR" -name "*.zip" -size 0 -delete 2>/dev/null || true

# ============================================================
# 3. Package manager caches
# ============================================================
log_info "Cleaning Python package manager caches..."

if command -v uv &> /dev/null; then
    export UV_CACHE_DIR="${TMP_CACHE_DIR}/uv"
    uv cache clean 2>/dev/null || true
    log_ok "uv cache cleaned."
fi

if command -v pip &> /dev/null; then
    pip cache purge 2>/dev/null || true
    log_ok "pip cache cleaned."
fi

# ============================================================
# 4. Conda cache
# ============================================================
if command -v conda &> /dev/null; then
    log_info "Cleaning Conda redundant packages..."
    conda clean --all -y > /dev/null 2>&1 || true
    log_ok "Conda cache cleaned."
fi

# ============================================================
# 5. APT cache
# ============================================================
log_info "Cleaning system-level APT cache..."
${APP_SUDO} apt-get clean -y 2>/dev/null || true
${APP_SUDO} apt-get autoclean -y 2>/dev/null || true
log_ok "APT cache cleaned."

# ============================================================
# 6. Python bytecode and Jupyter cache
# ============================================================
log_info "Removing project temporary files..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
log_ok "Temporary files removed."

# ============================================================
# 7. WandB temporary files
# ============================================================
if [ -d "$PROJECT_DIR/wandb" ]; then
    find "$PROJECT_DIR/wandb" -name "*.tmp" -delete 2>/dev/null || true
    log_ok "WandB temporary files cleaned."
fi

# ============================================================
# 8. Disk usage report
# ============================================================
echo ""
log_info "Disk usage report:"
echo "  Data disk ($PROJECT_PARENT_DIR):"
df -h "$PROJECT_PARENT_DIR" | awk 'NR==2 {print "    Used: " $3 " | Available: " $4 " | Usage: " $5}'
echo "  System disk (/):"
df -h / | awk 'NR==2 {print "    Used: " $3 " | Available: " $4 " | Usage: " $5}'

log_ok "Cache cleaning completed."