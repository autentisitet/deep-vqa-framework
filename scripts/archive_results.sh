#!/usr/bin/env bash
# --- archive_results.sh ---
#
# Purpose: Package Git-ignored heavy assets (results/ and datasets/)
# for backup or transfer.
#
# Usage: ./archive_results.sh [OPTIONS]

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
# Help
# ============================================================
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -r, --results     Package experimental outputs (results/)
  -d, --datasets    Package dataset files and caches (datasets/)
  -a, --all         Package both results/ and datasets/
  -h, --help        Show this help message

Examples:
  $(basename "$0") -r            # Archive results only
  $(basename "$0") -a            # Archive results + datasets
EOF
    exit 0
}

# ============================================================
# Parse arguments
# ============================================================
PACK_RESULTS=false
PACK_DATASETS=false

if [ $# -eq 0 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--results)   PACK_RESULTS=true; shift ;;
        -d|--datasets)  PACK_DATASETS=true; shift ;;
        -a|--all)       PACK_RESULTS=true; PACK_DATASETS=true; shift ;;
        -h|--help)      usage ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# ============================================================
# Paths
# ============================================================
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"
SAVE_DIR="$PROJECT_ROOT/archives"
TIMESTAMP=$(date +%m%d_%H%M)
SAVE_NAME="${SAVE_DIR}/PROJECT_ASSETS_${TIMESTAMP}.tar.gz"

mkdir -p "$SAVE_DIR"
cd "$PROJECT_ROOT" || exit 1

# ============================================================
# Collect targets
# ============================================================
PACK_LIST=()

if [ "$PACK_RESULTS" = true ]; then
    if [ -d "results" ]; then
        PACK_LIST+=("results")
    else
        log_warn "'results/' directory not found, skipping."
    fi
fi

if [ "$PACK_DATASETS" = true ]; then
    if [ -d "datasets" ]; then
        PACK_LIST+=("datasets")
    else
        log_warn "'datasets/' directory not found, skipping."
    fi
fi

if [ ${#PACK_LIST[@]} -eq 0 ]; then
    log_error "No valid directories found for packaging."
    exit 1
fi

# ============================================================
# Archive
# ============================================================
log_info "Packaging directories: ${PACK_LIST[*]}"
log_info "Output: $SAVE_NAME"

if tar -czf "$SAVE_NAME" "${PACK_LIST[@]}" 2>/dev/null; then
    SIZE=$(du -h "$SAVE_NAME" | cut -f1)
    log_ok "Archive created successfully."
    echo "  Path: $(CYAN)$SAVE_NAME$(RESET)"
    echo "  Size: $SIZE"
else
    log_error "Archive creation failed."
    exit 1
fi