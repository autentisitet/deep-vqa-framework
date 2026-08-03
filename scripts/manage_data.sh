#!/usr/bin/env bash
# --- manage_data.sh ---
#
# Purpose: Download, extract, and manage datasets for Deep-VQA-Framework
#
# Key tasks:
# 1. Search for datasets in public disk locations
# 2. Download missing datasets from official sources
# 3. Extract archives and create .done markers
# 4. Create symbolic links for easy access
#
# Usage: ./manage_data.sh

set -e -o pipefail

# ============================================================
# Color definitions (consistent with Makefile and setup_env.sh)
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
# Default values
# ============================================================
http_proxy="${http_proxy:-}"
https_proxy="${https_proxy:-}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
USER="${USER:-root}"

# ============================================================
# Project directories
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="${PROJECT_DIR}/datasets"
DOWNLOAD_CACHE="${PROJECT_DIR}/.download_cache"

# ============================================================
# Dataset paths
# ============================================================
TID_TARGET_PATH="${DATASETS_DIR}/TID2013"
KON_DATA_TARGET_PATH="${DATASETS_DIR}/KoNViD-1k/KoNViD-1k_videos"
KON_METADATA_TARGET_PATH="${DATASETS_DIR}/KoNViD-1k/KoNViD-1k_metadata"
# T2V_TARGET_PATH="${DATASETS_DIR}/T2VQA-DB"

# ============================================================
# Dataset URLs
# ============================================================
TID_SOURCE_URL="https://www.ponomarenko.info/tid2013/tid2013.rar"
KON_VIDEOS_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_videos.zip"
KON_METADATA_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_metadata.zip"
# T2V_SOURCE_URL="https://drive.google.com/file/d/1aak5hgYsXock19d1rVufss3_X6eEA4Wx/view"

# ============================================================
# Search directories (public disk)
# ============================================================
SEARCH_DIRS_=(
    "/root/autodl-pub/dataset"
    "/root/autodl-pub"
)

# ============================================================
# Helper: Check if dataset is valid (.done marker exists)
# ============================================================
is_dataset_valid() {
    local dir="$1"
    [ -f "$dir/.done" ]
}

# ============================================================
# Helper: Smart extract with flattening
# ============================================================
smart_extract() {
    local src_path="$1"
    local target_path="$2"

    if [ ! -f "$src_path" ]; then
        log_error "Source file '$src_path' not found."
        exit 1
    fi

    if [ -f "$target_path/.done" ]; then
        log_ok "$target_path already extracted (found .done marker)."
        return
    fi

    mkdir -p "$target_path"
    local temp_extract_dir="${target_path}_tmp_$(date +%s)"
    mkdir -p "$temp_extract_dir"
    log_info "Extracting $(basename "$src_path") to $target_path ..."

    if [[ "$src_path" == *.zip ]]; then
        unzip -q -o "$src_path" -d "$temp_extract_dir" || {
            log_error "Zip extraction failed"
            exit 1
        }
    elif [[ "$src_path" == *.rar ]]; then
        if command -v unrar &> /dev/null; then
            unrar x -o+ -y "$src_path" "$temp_extract_dir" > /dev/null || {
                log_error "Rar extraction failed"
                exit 1
            }
        else
            log_error "unrar not found. Please run: apt-get install unrar -y"
            exit 1
        fi
    else
        log_warn "Unknown archive format: $src_path"
        return 1
    fi

    # Flatten nested directories
    local content_count=$(ls -1 "$temp_extract_dir" | wc -l)
    local sub_dir=$(ls -1 "$temp_extract_dir")

    if [ "$content_count" -eq 1 ] && [ -d "$temp_extract_dir/$sub_dir" ]; then
        log_info "Detected nested folder '$sub_dir', flattening..."
        mv "$temp_extract_dir/$sub_dir"/* "$target_path/" 2>/dev/null || true
    else
        mv "$temp_extract_dir"/* "$target_path/" 2>/dev/null || true
    fi

    rm -rf "$temp_extract_dir"

    if [ -n "$USER" ]; then
        chown -R "$USER:$USER" "$target_path" 2>/dev/null || true
    fi
    chmod -R 755 "$target_path"

    touch "$target_path/.done"
    log_ok "Successfully extracted and fixed permissions for $target_path"
}

# ============================================================
# Helper: Search dataset in public disk
# ============================================================
search_dataset() {
    local keyword="$1"
    local search_paths=("${@:2}")

    # Priority 1: Search for directories with .done marker
    while IFS= read -r found_dir; do
        if [ -n "$found_dir" ] && [ -f "$found_dir/.done" ]; then
            echo "FOLDER|$found_dir"
            return
        fi
    done < <(find "${search_paths[@]}" -maxdepth 2 -type d -iname "*${keyword}*" 2>/dev/null)

    # Priority 2: Search for archive files
    local found_zip=$(find "${search_paths[@]}" \
        -maxdepth 2 \
        -type f \( -iname "*${keyword}*.zip" -o -iname "*${keyword}*.rar" \) \
        -print -quit 2>/dev/null)

    if [ -n "$found_zip" ]; then
        echo "ARCHIVE|$found_zip"
        return
    fi

    echo "NOT_FOUND|"
}

# ============================================================
# Helper: Handle dataset initialization
# ============================================================
handle_dataset_initialization() {
    local key="$1"
    local target="$2"
    local label="$3"

    log_info "Retrieving $label ..."

    # Check if already valid
    if is_dataset_valid "$target"; then
        log_ok "$label already exists and is valid at: $target"
        return 0
    elif [ -d "$target" ] && [ ! -f "$target/.done" ]; then
        log_warn "Directory exists but missing .done marker, checking public disk..."
        rm -rf "$target"
    fi

    # Search public disk
    local result=$(search_dataset "$key" "${SEARCH_DIRS_[@]}")
    local status="${result%%|*}"
    local found_path="${result#*|}"

    case "$status" in
        "FOLDER")
            log_ok "Found valid $label directory: $found_path"
            ln -snf "$found_path" "$target"
            log_info "Created symbolic link: $target -> $found_path"
            return 0
            ;;
        "ARCHIVE")
            log_ok "Found $label archive: $found_path"
            smart_extract "$found_path" "$target"
            return 0
            ;;
        *)
            log_warn "$label: No valid dataset found (requires a .done marker)."
            echo "    Checked: $target"
            echo "    If dataset is manually placed, run: touch $target/.done"
            return 1
            ;;
    esac
}

# ============================================================
# CI/CD guard
# ============================================================
if [ "${MANAGE_DATA_SOURCE_ONLY:-false}" = "true" ]; then
    return 0 2>/dev/null || exit 0
fi

# ============================================================
# Kill existing aria2c processes for this project
# ============================================================
if pgrep -f "aria2c.*${DOWNLOAD_CACHE}" > /dev/null; then
    log_warn "Found existing aria2c processes for this project, cleaning up..."
    pkill -f "aria2c.*${DOWNLOAD_CACHE}" || true
    sleep 1
fi

# ============================================================
# Dataset flags
# ============================================================
TID_DOWNLOAD_FLAG=false
KON_DATA_DOWNLOAD_FLAG=false
KON_METADATA_DOWNLOAD_FLAG=false
# T2V_DOWNLOAD_FLAG=false

# ============================================================
# Check each dataset
# ============================================================
handle_dataset_initialization "tid2013" "$TID_TARGET_PATH" "TID2013" || TID_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-videos" "$KON_DATA_TARGET_PATH" "KoNViD-1k" || KON_DATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-metadata" "$KON_METADATA_TARGET_PATH" "KoNViD-1k" || KON_METADATA_DOWNLOAD_FLAG=true
# handle_dataset_initialization "t2vqa-db" "$T2V_TARGET_PATH" "T2VQA-DB" || T2V_DOWNLOAD_FLAG=true

DOWNLOAD_FLAG=false
[ "$TID_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_DATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_METADATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
# [ "$T2V_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true

# ============================================================
# Proxy check
# ============================================================

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
# Download missing datasets
# ============================================================
if [ "$DOWNLOAD_FLAG" = true ]; then
    mkdir -p "$DOWNLOAD_CACHE"
    log_info "Some datasets are missing; starting downloads..."

    ulimit -n 65535
    log_info "Initiating sequential download mode for stability..."

    # Download KoNViD-1k videos
    if [ "$KON_DATA_DOWNLOAD_FLAG" = true ]; then
        log_info "Downloading KoNViD-1k videos dataset..."

        if [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" ]; then
            if ! unzip -t "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" &>/dev/null; then
                log_warn "Existing cache file is corrupted, removing..."
                rm -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip"
            fi
        fi

        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://datasets.vqa.mmsp-kn.de/" \
            --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
            --header="Accept-Language: en-US,en;q=0.9" \
            -x 16 -s 16 -k 1M \
            --min-split-size=1M \
            --max-connection-per-server=16 \
            --max-tries=99 \
            --retry-wait=10 \
            --timeout=60 \
            --console-log-level=notice \
            --summary-interval=2 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_videos.zip" \
            "${KON_VIDEOS_SOURCE_URL}" || log_warn "Failed to download KoNViD-1k videos"
    fi

    # Download KoNViD-1k metadata
    if [ "$KON_METADATA_DOWNLOAD_FLAG" = true ]; then
        log_info "Downloading KoNViD-1k metadata dataset..."
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://datasets.vqa.mmsp-kn.de/" \
            -x 16 -s 16 -k 1M \
            --min-split-size=1M \
            --max-connection-per-server=16 \
            --max-tries=99 \
            --retry-wait=10 \
            --timeout=60 \
            --console-log-level=notice \
            --summary-interval=2 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_metadata.zip" \
            "${KON_METADATA_SOURCE_URL}" || log_warn "Failed to download KoNViD-1k metadata"
    fi

    # Download TID2013
    if [ "$TID_DOWNLOAD_FLAG" = true ]; then
        log_info "Downloading TID2013 dataset..."

        if [ -f "${DOWNLOAD_CACHE}/tid2013.rar" ]; then
            if command -v unrar &> /dev/null; then
                if ! unrar t "${DOWNLOAD_CACHE}/tid2013.rar" &>/dev/null; then
                    log_warn "Existing cache file is corrupted, removing..."
                    rm -f "${DOWNLOAD_CACHE}/tid2013.rar"
                fi
            fi
        fi

        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://www.ponomarenko.info/tid2013/" \
            -x 16 -s 16 -k 1M \
            --min-split-size=1M \
            --max-connection-per-server=16 \
            --max-tries=99 \
            --retry-wait=10 \
            --timeout=60 \
            --console-log-level=notice \
            --summary-interval=2 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "tid2013.rar" \
            "${TID_SOURCE_URL}" || log_warn "Failed to download TID2013"
    fi

    # Download T2VQA
    # if [ "$T2V_DOWNLOAD_FLAG" = true ]; then
    #    log_info "Downloading T2VQA dataset..."
    #    if ! command -v gdown &> /dev/null; then
    #        uv lock --upgrade-package gdown 2>/dev/null || true
    #        uv run gdown --version 2>/dev/null || true
    #    fi
    #    uv run gdown -user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
    #        --continue \
    #        "${T2V_SOURCE_URL}" \
    #        -O "${DOWNLOAD_CACHE}/t2vqa.zip" 2>/dev/null || log_warn "Failed to download T2VQA"
    # fi

    log_ok "All dataset downloads completed."

    # Extract downloaded datasets
    log_info "Extracting downloaded datasets..."
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" "$KON_DATA_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/tid2013.rar" ] && smart_extract "${DOWNLOAD_CACHE}/tid2013.rar" "$TID_TARGET_PATH"
    # [ -f "${DOWNLOAD_CACHE}/t2vqa.zip" ] && smart_extract "${DOWNLOAD_CACHE}/t2vqa.zip" "$T2V_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" "$KON_METADATA_TARGET_PATH"
fi

# ============================================================
# Validation
# ============================================================
for dir in "$TID_TARGET_PATH" "$KON_DATA_TARGET_PATH" "$KON_METADATA_TARGET_PATH"; do # "$T2V_TARGET_PATH"
    if is_dataset_valid "$dir"; then
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l)
        log_ok "$dir: $file_count files found"
    else
        log_warn "$dir is missing or invalid (no .done marker)"
    fi
done

log_ok "Dataset preparation completed."

# ============================================================
# Save download flag for CI/CD
# ============================================================
mkdir -p "${PROJECT_DIR}/results/scripts_logs"
echo "DOWNLOAD_FLAG=$DOWNLOAD_FLAG" > "${PROJECT_DIR}/results/scripts_logs/.download_flag"

# ============================================================
# Create symbolic links
# ============================================================
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"
[ -d "TID2013" ] && ln -snf TID2013 tid2013
[ -d "KoNViD-1k" ] && ln -snf KoNViD-1k konvid-1k
# [ -d "T2VQA-DB" ] && ln -snf T2VQA-DB t2vqa-db

log_info "Current symbolic links:"
ls -la | grep "^l" || echo "  (none)"