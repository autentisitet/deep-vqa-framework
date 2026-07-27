#!/usr/bin/env bash
# --- archive_results.sh ---
# Description: Minimalist & Robust Archiving Tool
# Packages Git-ignored heavy assets: results/ and datasets/

set -e -o pipefail

# --- 1. Physical path auto-detection ---
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_NAME=$(basename "$0")

PROJECT_ROOT=$(cd "$BASE_DIR/.."; pwd)
SAVE_DIR="$PROJECT_ROOT/archives"
TIMESTAMP=$(date +%m%d_%H%M)
SAVE_NAME="${SAVE_DIR}/PROJECT_ASSETS_${TIMESTAMP}.tar.gz"

mkdir -p "$SAVE_DIR"

# Default flags
PACK_RESULTS=false
PACK_DATASETS=false

usage() {
    echo "Usage: $SCRIPT_NAME [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --results     Package all experimental outputs (results/)"
    echo "  -d, --datasets    Package dataset files and caches (datasets/)"
    echo "  -a, --all         Package both results/ and datasets/"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $SCRIPT_NAME -r            # Quick archive: experimental results only"
    echo "  $SCRIPT_NAME -a            # Complete archive: results + datasets"
    exit 0
}

if [ $# -eq 0 ]; then
    usage
fi

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--results)
            PACK_RESULTS=true
            shift
            ;;
        -d|--datasets)
            PACK_DATASETS=true
            shift
            ;;
        -a|--all)
            PACK_RESULTS=true
            PACK_DATASETS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

cd "$PROJECT_ROOT" || exit 1

PACK_LIST=()

# Collect target directories
if [ "$PACK_RESULTS" = true ]; then
    if [ -d "results" ]; then
        PACK_LIST+=("results")
    else
        echo "⚠️  Warning: 'results/' directory not found, skipping."
    fi
fi

if [ "$PACK_DATASETS" = true ]; then
    if [ -d "datasets" ]; then
        PACK_LIST+=("datasets")
    else
        echo "⚠️  Warning: 'datasets/' directory not found, skipping."
    fi
fi

# Empty asset guard
if [ ${#PACK_LIST[@]} -eq 0 ]; then
    echo "❌ Error: No valid directories found for packaging!"
    exit 1
fi

echo "=========================================="
echo "📦 Packaging directories: ${PACK_LIST[*]}"
echo "📂 Project root: $PROJECT_ROOT"
echo "🚀 Output file: $SAVE_NAME"
echo "=========================================="

# Perform tar compression
tar -czf "$SAVE_NAME" "${PACK_LIST[@]}"

if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "✅ Archive successful -> $SAVE_NAME"
    echo "📊 Package size: $(du -h "$SAVE_NAME" | cut -f1)"
    echo "------------------------------------------"
else
    echo "❌ Archive failed."
    exit 1
fi