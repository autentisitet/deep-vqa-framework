#!/usr/bin/env bash
# --- manage_data.sh ---

set -e -o pipefail

# Set default values to prevent undefined variables.
http_proxy="${http_proxy:-}"
https_proxy="${https_proxy:-}"
USER="${USER:-root}"


smart_extract() {
    local src_path="$1"
    local target_path="$2"

    if [ ! -f "$src_path" ]; then
        echo "[X] Error: Source file '$src_path' not found. Skip extraction."
        exit 1
    fi

    if [ -d "$target_path" ] && [ "$(ls -A "$target_path")" ]; then
        echo "[√] $target_path already exists; skip decompression."
        return
    fi

    mkdir -p "$target_path"
    local temp_extract_dir="${target_path}_tmp_$(date +%s)"
    mkdir -p "$temp_extract_dir"
    echo "📂 Extracting from $(basename "$src_path") to $target_path ..."

    if [[ "$src_path" == *.zip ]]; then
        unzip -q -o "$src_path" -d "$temp_extract_dir" || { echo "Zip extraction failed"; exit 1; }
    elif [[ "$src_path" == *.rar ]]; then
        if command -v unrar &> /dev/null; then
            unrar x -o+ -y "$src_path" "$temp_extract_dir" > /dev/null || { echo "Rar extraction failed"; exit 1; }
        else
            echo "[X] Error: unrar not found, please execute apt-get install unrar -y"
            exit 1
        fi
    fi

    local content_count=$(ls -1 "$temp_extract_dir" | wc -l)
    local sub_dir=$(ls -1 "$temp_extract_dir")

    if [ "$content_count" -eq 1 ] && [ -d "$temp_extract_dir/$sub_dir" ]; then
        echo "📦 Detected nested folder '$sub_dir', flattening..."
        mv "$temp_extract_dir/$sub_dir"/* "$target_path/" 2>/dev/null || true
    else
        mv "$temp_extract_dir"/* "$target_path/" 2>/dev/null || true
    fi

    rm -rf "$temp_extract_dir"

    if [ -n "$USER" ]; then
        chown -R "$USER:$USER" "$target_path" 2>/dev/null || true
    fi
    chmod -R 755 "$target_path"

    echo -e "\033[1;34m✨ Successfully extracted and fixed permissions for $target_path\033[0m"
}


proxy_on() {
    local port="${1:-7890}"
    export http_proxy="http://127.0.0.1:${port}"
    export https_proxy="${http_proxy}"
    echo "Proxy enabled on port ${port}"
}


detect_proxy_port() {
    for port in 7890 7897 10809 1080 22741; do
        if curl -s -o /dev/null --max-time 2 --proxy "http://127.0.0.1:$port" "https://www.baidu.com" 2>/dev/null; then
            echo "$port"
            return 0
        fi
    done

    return 1
}


# Target Dataset: TID2013, KoNViD-1k, T2VQA-DB
TID_SOURCE_URL="https://www.ponomarenko.info/tid2013/tid2013.rar"
KON_VIDEOS_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_videos.zip"
KON_METADATA_SOURCE_URL="https://datasets.vqa.mmsp-kn.de/archives/KoNViD_1k_metadata.zip"
T2V_SOURCE_URL="https://drive.google.com/file/d/1aak5hgYsXock19d1rVufss3_X6eEA4Wx/view"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="${PROJECT_DIR}/datasets"

SEARCH_DIRS_=(
    "/root/autodl-pub/dataset"
    "/root/autodl-pub"
)

TID_DOWNLOAD_FLAG=false
KON_DATA_DOWNLOAD_FLAG=false
KON_METADATA_DOWNLOAD_FLAG=false
T2V_DOWNLOAD_FLAG=false

TID_TARGET_PATH="${PROJECT_DIR}/datasets/TID2013"
KON_DATA_TARGET_PATH="${PROJECT_DIR}/datasets/KoNViD-1k/KoNViD-1k_videos"
KON_METADATA_TARGET_PATH="${PROJECT_DIR}/datasets/KoNViD-1k/KoNViD-1k_metadata"
T2V_TARGET_PATH="${PROJECT_DIR}/datasets/T2VQA-DB"

DOWNLOAD_CACHE="${PROJECT_DIR}/.download_cache"


search_dataset() {
    local keyword="$1"
    local search_paths=("${@:2}")

    local found_dir=$(find "${search_paths[@]}" \
                            -maxdepth 4 \
                            -type d \
                            -iname "*${keyword}*" \
                            -print -quit 2>/dev/null)

    if [ -n "$found_dir" ]; then
        echo "FOLDER|$found_dir"
        return
    fi

    local found_zip=$(find "${search_paths[@]}" \
                            -maxdepth 4 \
                            -type f \( -iname "*${keyword}*.zip" \
                                -o -iname "*${keyword}*.rar" \) \
                            -print -quit 2>/dev/null)

    if [ -n "$found_zip" ]; then
        echo "ARCHIVE|$found_zip"
        return
    fi

    echo "NOT_FOUND|"
}


# Check whether the directory contains valid data (i.e., is not an empty directory).
is_dataset_valid() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return 1
    fi

    if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 1
    fi

    local found=$(find "$dir" -maxdepth 1 -type f \( \
        -iname "*.png" -o \
        -iname "*.PNG" -o \
        -iname "*.jpg" -o \
        -iname "*.JPG" -o \
        -iname "*.jpeg" -o \
        -iname "*.JPEG" -o \
        -iname "*.bmp" -o \
        -iname "*.BMP" -o \
        -iname "*.tif" -o \
        -iname "*.tiff" -o \
        -iname "*.TIF" -o \
        -iname "*.TIFF" -o \
        -iname "*.webp" -o \
        -iname "*.WEBP" -o \
        -iname "*.ppm" -o \
        -iname "*.PPM" -o \
        -iname "*.mp4" -o \
        -iname "*.MP4" -o \
        -iname "*.avi" -o \
        -iname "*.AVI" -o \
        -iname "*.mov" -o \
        -iname "*.MOV" -o \
        -iname "*.mkv" -o \
        -iname "*.MKV" -o \
        -iname "*.webm" -o \
        -iname "*.WEBM" \
    \) 2>/dev/null | head -1)

    if [ -n "$found" ]; then
        return 0
    fi

    return 1
}


handle_dataset_initialization() {
    local key="$1"
    local target="$2"
    local label="$3"

    echo "🔍 Retrieving $label ..."

    # 1. First, check whether the target path already contains data.
    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "[√] $label already exists in datasets directory: $target"
        # Check for the existence of actual data files.
        local data_files=$(find "$target" -type f 2>/dev/null | head -1)
        if [ -n "$data_files" ]; then
            echo "    -> Found data files, using existing dataset"
            return 0
        else
            echo "    ⚠️  Directory exists but appears empty, checking other sources..."
        fi
    fi

    # 2. If there is no data at the target path, search the public directory.
    local result=$(search_dataset "$key" "${SEARCH_DIRS_[@]}")
    local status="${result%%|*}"
    local found_path="${result#*|}"

    case "$status" in
        "FOLDER")
            echo "[√] The $label directory was found: $found_path"

            # If the target path already exists but is empty, delete it first.
            if [ -d "$target" ] && [ -z "$(ls -A "$target" 2>/dev/null)" ]; then
                rmdir "$target" 2>/dev/null || true
            fi
            ln -snf "$found_path" "$target"
            echo "    -> A symbolic link has been created to $target"
            return 0
            ;;
        "ARCHIVE")
            echo "[√] The $label compressed file was found: $found_path"
            smart_extract "$found_path" "$target"
            echo "    -> Decompressed in place $target"
            return 0
            ;;
        *)
            echo "[!] Not in the local area or public $label"
            return 1
            ;;
    esac
}


handle_dataset_initialization "tid2013" "$TID_TARGET_PATH" "TID2013" || TID_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-videos" "$KON_DATA_TARGET_PATH" "KoNViD-1k" || KON_DATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-metadata" "$KON_METADATA_TARGET_PATH" "KoNViD-1k" || KON_METADATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "t2vqa-db" "$T2V_TARGET_PATH" "T2VQA-DB" || T2V_DOWNLOAD_FLAG=true

DOWNLOAD_FLAG=false
[ "$TID_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_DATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_METADATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$T2V_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true


# --- 代理处理核心修复点 ---
if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
    echo "✅ Using existing environment proxy: ${http_proxy:-$HTTP_PROXY}"
else
    PROXY_PORT=$(detect_proxy_port 2>/dev/null) || true
    if [ -n "$PROXY_PORT" ]; then
        proxy_on "$PROXY_PORT"
    else
        echo "ℹ️  No active proxy detected, proceeding with direct connection."
    fi
fi


if [ "$DOWNLOAD_FLAG" = true ]; then
    mkdir -p "$DOWNLOAD_CACHE"
    echo "[!] Some datasets are missing; download mode is now available."

    ulimit -n 65535
    echo "Initiating sequential download mode for stability..."

    if [ "$KON_DATA_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the konvid-1k videos dataset..."

        if [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" ]; then
            if ! unzip -t "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" &>/dev/null; then
                echo "⚠️  Existing cache file is corrupted, removing..."
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
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_videos.zip" \
            "${KON_VIDEOS_SOURCE_URL}" || echo "⚠️  Failed to download KoNViD-1k videos"
    fi

    if [ "$KON_METADATA_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the konvid-1k metadata dataset..."
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://datasets.vqa.mmsp-kn.de/" \
            -x 16 -s 16 -k 1M \
            --min-split-size=1M \
            --max-connection-per-server=16 \
            --max-tries=5 --retry-wait=5 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "KoNViD_1k_metadata.zip" \
            "${KON_METADATA_SOURCE_URL}" || echo "⚠️  Failed to download KoNViD-1k metadata"
    fi

    if [ "$TID_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the tid2013 dataset..."
        aria2c --check-certificate=false \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
            --header="Referer: https://www.ponomarenko.info/tid2013/" \
            -x 16 -s 16 -k 1M \
            --min-split-size=1M \
            --max-connection-per-server=16 \
            --max-tries=5 --retry-wait=5 \
            -c \
            -d "$DOWNLOAD_CACHE" \
            -o "tid2013.rar" \
            "${TID_SOURCE_URL}" || echo "⚠️  Failed to download TID2013"
    fi

    if [ "$T2V_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the t2vqa dataset..."
        if ! command -v gdown &> /dev/null; then
            uv lock --upgrade-package gdown 2>/dev/null || true
            uv run gdown --version 2>/dev/null || true
        fi
        uv run gdown -O "${DOWNLOAD_CACHE}/t2vqa.zip" \
            --continue \
            "${T2V_SOURCE_URL}" 2>/dev/null || echo "⚠️  Failed to download T2VQA"
    fi

    echo "All dataset downloads have been completed."

    echo "📦 Extracting downloaded datasets..."
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" "$KON_DATA_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/tid2013.rar" ] && smart_extract "${DOWNLOAD_CACHE}/tid2013.rar" "$TID_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/t2vqa.zip" ] && smart_extract "${DOWNLOAD_CACHE}/t2vqa.zip" "$T2V_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" "$KON_METADATA_TARGET_PATH"
fi

# Validation (warns if anomalies are detected, but does not force an exit)
for dir in "$TID_TARGET_PATH" "$KON_DATA_TARGET_PATH" "$KON_METADATA_TARGET_PATH" "$T2V_TARGET_PATH"; do
    if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "⚠️  $dir is empty or missing"
    else
        # Check for the existence of actual data files.
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l)
        if [ "$file_count" -eq 0 ]; then
            echo "⚠️  $dir exists but contains no files (may be empty directory)"
        else
            echo "✅ $dir: $file_count files found"
        fi
    fi
done

echo "Dataset preparation completed."
mkdir -p "${PROJECT_DIR}/results/scripts_logs"
echo "DOWNLOAD_FLAG=$DOWNLOAD_FLAG" > "${PROJECT_DIR}/results/scripts_logs/.download_flag"

cd "$DATASETS_DIR"
ln -snf TID2013 tid2013
ln -snf KoNViD-1k konvid-1k
ln -snf T2VQA-DB t2vqa-db

ls -la | grep "^l"