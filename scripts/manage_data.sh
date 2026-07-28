#!/usr/bin/env bash
# --- manage_data.sh ---

set -e -o pipefail

# Set default values to prevent undefined variables.
http_proxy="${http_proxy:-}"
https_proxy="${https_proxy:-}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
USER="${USER:-root}"


smart_extract() {
    local src_path="$1"
    local target_path="$2"

    if [ ! -f "$src_path" ]; then
        echo "[X] Error: Source file '$src_path' not found. Skip extraction."
        exit 1
    fi

    # 如果目录存在且有 .done 标记，跳过解压
    if [ -f "$target_path/.done" ]; then
        echo "[√] $target_path already extracted (found .done marker)."
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

    # 解压成功，创建 .done 标记
    touch "$target_path/.done"

    echo -e "\033[1;34m✨ Successfully extracted and fixed permissions for $target_path\033[0m"
}


# Check whether the directory contains valid data (i.e., .done marker exists).
is_dataset_valid() {
    local dir="$1"
    # 目录存在且 .done 文件存在 → 有效
    [ -f "$dir/.done" ]
}


search_dataset() {
    local keyword="$1"
    local search_paths=("${@:2}")

    # 1. 优先搜索文件夹，并校验里面是否有 .done 标记
    while IFS= read -r found_dir; do
        if [ -n "$found_dir" ] && [ -f "$found_dir/.done" ]; then
            echo "FOLDER|$found_dir"
            return
        fi
    done < <(find "${search_paths[@]}" -maxdepth 2 -type d -iname "*${keyword}*" 2>/dev/null)

    # 2. 再搜索压缩包
    local found_zip=$(find "${search_paths[@]}" \
                            -maxdepth 2 \
                            -type f \( -iname "*${keyword}*.zip" \
                                -o -iname "*${keyword}*.rar" \) \
                            -print -quit 2>/dev/null)

    if [ -n "$found_zip" ]; then
        echo "ARCHIVE|$found_zip"
        return
    fi

    echo "NOT_FOUND|"
}


# Target Dataset URLs
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


handle_dataset_initialization() {
    local key="$1"
    local target="$2"
    local label="$3"

    echo "🔍 Retrieving $label ..."

    # 1. 校验目标目录是否已经存在且有效（有 .done 标记）
    if is_dataset_valid "$target"; then
        echo "[√] $label already exists and is valid at: $target"
        return 0
    elif [ -d "$target" ] && [ ! -f "$target/.done" ]; then
        echo "    ⚠️ Directory exists but missing .done marker, checking public disk..."
        # 删除没有 .done 标记的目录，重新处理
        rm -rf "$target"
    fi

    # 2. 检索公共盘
    local result=$(search_dataset "$key" "${SEARCH_DIRS_[@]}")
    local status="${result%%|*}"
    local found_path="${result#*|}"

    case "$status" in
        "FOLDER")
            echo "[√] Found valid $label directory: $found_path"
            ln -snf "$found_path" "$target"
            echo "    -> Created symbolic link: $target -> $found_path"
            return 0
            ;;
        "ARCHIVE")
            echo "[√] Found $label archive: $found_path"
            smart_extract "$found_path" "$target"
            return 0
            ;;
        *)
            echo "⚠️  $label: No valid dataset found (requires a .done marker)."
            echo "    Checked: $target"
            echo "    If the dataset is manually placed, run: touch $target/.done"
            echo "    Otherwise, the download will be triggered."
            return 1
            ;;
    esac
}

# 避免 CI 时完整执行 manage_data.sh
if [ "${MANAGE_DATA_SOURCE_ONLY:-false}" = "true" ]; then
    return 0 2>/dev/null || exit 0
fi


# 只杀掉正在下载到当前项目 DOWNLOAD_CACHE 目录的 aria2c 进程
if pgrep -f "aria2c.*${DOWNLOAD_CACHE}" > /dev/null; then
    echo "⚠️ Found existing aria2c processes for this project, cleaning up..."
    pkill -f "aria2c.*${DOWNLOAD_CACHE}" || true
    sleep 1
fi


handle_dataset_initialization "tid2013" "$TID_TARGET_PATH" "TID2013" || TID_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-videos" "$KON_DATA_TARGET_PATH" "KoNViD-1k" || KON_DATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "konvid-1k-metadata" "$KON_METADATA_TARGET_PATH" "KoNViD-1k" || KON_METADATA_DOWNLOAD_FLAG=true
handle_dataset_initialization "t2vqa-db" "$T2V_TARGET_PATH" "T2VQA-DB" || T2V_DOWNLOAD_FLAG=true

DOWNLOAD_FLAG=false
[ "$TID_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_DATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$KON_METADATA_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true
[ "$T2V_DOWNLOAD_FLAG" = true ] && DOWNLOAD_FLAG=true

# 干净的网络代理策略：完全尊重当前终端/环境配置，不盲猜端口
if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
    echo "✅ Using environment proxy: ${http_proxy:-$HTTP_PROXY}"
else
    echo "ℹ️ No proxy environment variables set. Proceeding with direct connection."
    echo "   (Tip: Run 'source /etc/network_turbo' first if downloading on AutoDL)"
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
                echo "⚠️ Existing cache file is corrupted, removing..."
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
            "${KON_VIDEOS_SOURCE_URL}" || echo "⚠️ Failed to download KoNViD-1k videos"
    fi

    if [ "$KON_METADATA_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the konvid-1k metadata dataset..."
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
            "${KON_METADATA_SOURCE_URL}" || echo "⚠️ Failed to download KoNViD-1k metadata"
    fi

    if [ "$TID_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the tid2013 dataset..."

        if [ -f "${DOWNLOAD_CACHE}/tid2013.rar" ]; then
            if command -v unrar &> /dev/null; then
                if ! unrar t "${DOWNLOAD_CACHE}/tid2013.rar" &>/dev/null; then
                    echo "⚠️ Existing cache file is corrupted, removing..."
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
            "${TID_SOURCE_URL}" || echo "⚠️ Failed to download TID2013"
    fi

    if [ "$T2V_DOWNLOAD_FLAG" = true ]; then
        echo "Downloading the t2vqa dataset..."
        if ! command -v gdown &> /dev/null; then
            uv lock --upgrade-package gdown 2>/dev/null || true
            uv run gdown --version 2>/dev/null || true
        fi
        uv run gdown -O "${DOWNLOAD_CACHE}/t2vqa.zip" \
            --continue \
            "${T2V_SOURCE_URL}" 2>/dev/null || echo "⚠️ Failed to download T2VQA"
    fi

    echo "All dataset downloads have been completed."

    echo "📦 Extracting downloaded datasets..."
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_videos.zip" "$KON_DATA_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/tid2013.rar" ] && smart_extract "${DOWNLOAD_CACHE}/tid2013.rar" "$TID_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/t2vqa.zip" ] && smart_extract "${DOWNLOAD_CACHE}/t2vqa.zip" "$T2V_TARGET_PATH"
    [ -f "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" ] && smart_extract "${DOWNLOAD_CACHE}/KoNViD_1k_metadata.zip" "$KON_METADATA_TARGET_PATH"
fi

# Validation
for dir in "$TID_TARGET_PATH" "$KON_DATA_TARGET_PATH" "$KON_METADATA_TARGET_PATH" "$T2V_TARGET_PATH"; do
    if is_dataset_valid "$dir"; then
        file_count=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo "✅ $dir: $file_count files found"
    else
        echo "⚠️  $dir is missing or invalid (no .done marker)"
    fi
done

echo "Dataset preparation completed."
mkdir -p "${PROJECT_DIR}/results/scripts_logs"
echo "DOWNLOAD_FLAG=$DOWNLOAD_FLAG" > "${PROJECT_DIR}/results/scripts_logs/.download_flag"


mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"
[ -d "TID2013" ] && ln -snf TID2013 tid2013
[ -d "KoNViD-1k" ] && ln -snf KoNViD-1k konvid-1k
[ -d "T2VQA-DB" ] && ln -snf T2VQA-DB t2vqa-db

echo "Current symbolic links:"
ls -la | grep "^l" || true