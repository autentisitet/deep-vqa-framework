#!/usr/bin/env bash
# --- system_check.sh ---
# Deep-VQA System Health Check Script
# Linux Environment Diagnostic Tool
#
# Coverage:
# 1. Training Processes & FD Limit   7. Conda / Micromamba Envs
# 2. Ports (TB, Jupyter, Gradio)     8. Fast UV Package Manager
# 3. System Memory (RAM)             9. Network & PyPI/APT Mirrors
# 4. GPU Memory & Thermals          10. User Permissions & Disk Space
# 5. Storage Space (Root/Tmp)       11. PyTorch & Transformers Stack
# 6. PIP, Pyenv & Venv Status       12. Jupyter & Service Configurations

set -e -o pipefail

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ------------------------------------------------------------
# Dependency check (non-fatal)
# ------------------------------------------------------------
check_dependencies() {
    local missing=()
    for cmd in bc curl nvidia-smi; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${YELLOW}[!] Optional dependencies missing: ${missing[*]}${NC}"
        echo -e "${YELLOW}    Some checks may be incomplete.${NC}"
    fi
}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}=== Deep-VQA System Health Check ===${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "Host: $(hostname 2>/dev/null || echo 'unknown')"
check_dependencies

# ------------------------------------------------------------
# 1. Process check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[1/12] Checking Python training processes...${NC}"
TRAIN_PROCESS=$(ps -ef 2>/dev/null | grep -E "main.py|train.py" | grep -v "grep" | grep -v "system_check" || true)
if [ -z "$TRAIN_PROCESS" ]; then
    echo -e "${RED}[X] No running training process found.${NC}"
else
    echo -e "${GREEN}[√] Running training processes:${NC}"
    echo "$TRAIN_PROCESS" | awk '{print "  PID: " $2 " | CMD: " $8 " " $9 " " $10 " " $11 " " $12}'
fi

# File descriptor limit
echo -e "  File descriptor limit: $(ulimit -n 2>/dev/null || echo 'N/A')"

# ------------------------------------------------------------
# 2. Port check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[2/12] Checking common ports...${NC}"
COMMON_PORTS=(6006 8888 8080 8000 22 7860)
for port in "${COMMON_PORTS[@]}"; do
    if command -v ss &> /dev/null; then
        PORT_CHECK=$(ss -tlnp 2>/dev/null | grep -q ":$port " && echo "LISTENING" || echo "FREE")
    elif command -v netstat &> /dev/null; then
        PORT_CHECK=$(netstat -tlnp 2>/dev/null | grep -q ":$port " && echo "LISTENING" || echo "FREE")
    else
        PORT_CHECK="UNKNOWN (install ss or netstat)"
    fi
    if [ "$PORT_CHECK" = "LISTENING" ]; then
        echo -e "  Port ${port}: ${GREEN}LISTENING${NC}"
    elif [ "$PORT_CHECK" = "FREE" ]; then
        echo -e "  Port ${port}: ${YELLOW}FREE${NC}"
    else
        echo -e "  Port ${port}: ${YELLOW}UNKNOWN${NC}"
    fi
done

# ------------------------------------------------------------
# 3. Memory check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[3/12] Memory usage status:${NC}"
if command -v free &> /dev/null; then
    MEM_TOTAL=$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}')
    MEM_USED=$(free -h 2>/dev/null | awk '/^Mem:/ {print $3}')
    MEM_AVAIL=$(free -h 2>/dev/null | awk '/^Mem:/ {print $7}')
    MEM_PERCENT=$(free 2>/dev/null | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
    if command -v bc &> /dev/null; then
        if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
            MEM_COLOR=$RED
        elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
            MEM_COLOR=$YELLOW
        else
            MEM_COLOR=$GREEN
        fi
    else
        MEM_COLOR=$YELLOW
    fi
    echo -e "  Total: ${MEM_TOTAL} | Used: ${MEM_USED} | Available: ${MEM_AVAIL} | Usage: ${MEM_COLOR}${MEM_PERCENT}%${NC}"
else
    echo -e "  ${YELLOW}[!] free command not available${NC}"
fi

# ------------------------------------------------------------
# 4. GPU memory check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[4/12] GPU memory status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi -L &> /dev/null; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r gpu_id gpu_util mem_used mem_total gpu_temp; do
            if command -v bc &> /dev/null && [ -n "$mem_used" ] && [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ]; then
                MEM_PERCENT=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
                if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
                    MEM_COLOR=$RED
                elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
                    MEM_COLOR=$YELLOW
                else
                    MEM_COLOR=$GREEN
                fi
            else
                MEM_PERCENT="N/A"
                MEM_COLOR=$YELLOW
            fi
            echo -e "  GPU $gpu_id: Util ${gpu_util}% | Memory ${mem_used}MiB / ${mem_total}MiB (${MEM_COLOR}${MEM_PERCENT}%${NC}) | Temp ${gpu_temp}°C"
        done
    else
        echo -e "  ${YELLOW}[!] No GPU detected or nvidia-smi failed.${NC}"
    fi
else
    echo -e "  ${YELLOW}[!] nvidia-smi not found (no GPU or driver not installed)${NC}"
fi

# ------------------------------------------------------------
# 5. Disk space check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[5/12] Disk space usage:${NC}"
if command -v df &> /dev/null; then
    df -h / /root/autodl-tmp 2>/dev/null | awk 'NR==1 {print "  Filesystem      Size  Used  Avail  Use% Mount"} NR>1 {printf "  %-15s %-5s %-5s %-5s %-4s %s\n", $1, $2, $3, $4, $5, $6}'
else
    echo -e "  ${YELLOW}[!] df command not available${NC}"
fi

# ------------------------------------------------------------
# 6. Python environment managers
# ------------------------------------------------------------
echo -e "\n${YELLOW}[6/12] Python environment managers:${NC}"
for tool in pip pyenv virtualenv; do
    if command -v "$tool" &> /dev/null; then
        VERSION=$("$tool" --version 2>/dev/null | head -1)
        echo -e "  ${GREEN}[√] $tool: $VERSION${NC}"
    else
        echo -e "  ${YELLOW}[!] $tool not found${NC}"
    fi
done
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "  ${GREEN}[√] Active venv: $VIRTUAL_ENV${NC}"
else
    echo -e "  ${YELLOW}[!] No active virtual environment detected${NC}"
fi

# ------------------------------------------------------------
# 7. Conda / Mamba
# ------------------------------------------------------------
echo -e "\n${YELLOW}[7/12] Conda/Mamba environments:${NC}"
if command -v conda &> /dev/null; then
    CONDA_ENV=$(conda info --envs 2>/dev/null | grep '\*' | awk '{print $1}' || echo "base")
    CONDA_PYTHON=$(conda info 2>/dev/null | grep "python version" | awk '{print $3}')
    echo -e "  ${GREEN}[√] Conda found (active: $CONDA_ENV, python: $CONDA_PYTHON)${NC}"
else
    echo -e "  ${YELLOW}[!] Conda not found${NC}"
fi
if command -v micromamba &> /dev/null; then
    MAMBA_VERSION=$(micromamba --version 2>/dev/null)
    echo -e "  ${GREEN}[√] Micromamba: $MAMBA_VERSION${NC}"
else
    echo -e "  ${YELLOW}[!] Micromamba not found${NC}"
fi

# ------------------------------------------------------------
# 8. UV package manager
# ------------------------------------------------------------
echo -e "\n${YELLOW}[8/12] UV package manager:${NC}"
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>/dev/null)
    echo -e "  ${GREEN}[√] $UV_VERSION${NC}"
else
    echo -e "  ${YELLOW}[!] UV not found (not installed)${NC}"
fi

# ------------------------------------------------------------
# 9. Network & mirrors
# ------------------------------------------------------------
echo -e "\n${YELLOW}[9/12] Network connectivity check:${NC}"
# Internet connectivity
if curl -s -I --max-time 3 "https://www.google.com" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|301|302|307"; then
    echo -e "  ${GREEN}[√] Internet connectivity: OK${NC}"
elif curl -s -I --max-time 3 "https://www.baidu.com" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|301|302"; then
    echo -e "  ${GREEN}[√] Internet connectivity: OK (via baidu)${NC}"
else
    echo -e "  ${RED}[X] Internet connectivity: FAILED${NC}"
fi
# Proxy status
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    echo -e "  ${YELLOW}[!] Proxy detected: http_proxy=${http_proxy:-unset}, https_proxy=${https_proxy:-unset}${NC}"
else
    echo -e "  ${GREEN}[√] No proxy configured${NC}"
fi
# APT mirror
if [ -f /etc/apt/sources.list ]; then
    APT_MIRROR=$(grep -v "^#" /etc/apt/sources.list 2>/dev/null | head -1 | awk '{print $2}' | cut -d'/' -f3 2>/dev/null || echo "N/A")
    echo -e "  ${GREEN}[√] APT mirror: $APT_MIRROR${NC}"
fi
# pip mirror (check config)
if [ -f ~/.pip/pip.conf ]; then
    PIP_MIRROR=$(grep -E "^index-url|^extra-index-url" ~/.pip/pip.conf 2>/dev/null | head -1 | awk '{print $3}' || echo "N/A")
    echo -e "  ${GREEN}[√] pip mirror configured: $PIP_MIRROR${NC}"
fi

# ------------------------------------------------------------
# 10. Permission check
# ------------------------------------------------------------
echo -e "\n${YELLOW}[10/12] Permission checks:${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || echo '.')"
if [ -w "$SCRIPT_DIR" ]; then
    echo -e "  ${GREEN}[√] Write permission: $SCRIPT_DIR${NC}"
else
    echo -e "  ${RED}[X] No write permission: $SCRIPT_DIR${NC}"
fi
if [ "$EUID" -eq 0 ]; then
    echo -e "  ${YELLOW}[!] Running as root (use with caution)${NC}"
else
    echo -e "  ${GREEN}[√] Running as non-root user: ${USER:-unknown}${NC}"
fi
# Home directory space
if [ -d "$HOME" ]; then
    HOME_USAGE=$(du -sh "$HOME" 2>/dev/null | awk '{print $1}' || echo "N/A")
    echo -e "  Home directory size: $HOME_USAGE"
fi

# ------------------------------------------------------------
# 11. Python interpreter
# ------------------------------------------------------------
echo -e "\n${YELLOW}[11/12] Python interpreter:${NC}"
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi
if [ -n "$PYTHON_CMD" ]; then
    PYTHON_PATH=$(which "$PYTHON_CMD" 2>/dev/null)
    PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
    echo -e "  ${GREEN}[√] Python: $PYTHON_VERSION ($PYTHON_PATH)${NC}"
    # Check key packages
    echo -e "  ${CYAN}    Key packages:${NC}"
    "$PYTHON_CMD" -c "import torch; print(f'    torch: {torch.__version__}')" 2>/dev/null || echo -e "    ${YELLOW}torch: NOT FOUND${NC}"
    "$PYTHON_CMD" -c "import torchvision; print(f'    torchvision: {torchvision.__version__}')" 2>/dev/null || echo -e "    ${YELLOW}torchvision: NOT FOUND${NC}"
    "$PYTHON_CMD" -c "import transformers; print(f'    transformers: {transformers.__version__}')" 2>/dev/null || echo -e "    ${YELLOW}transformers: NOT FOUND${NC}"
    "$PYTHON_CMD" -c "import fastapi; print(f'    fastapi: {fastapi.__version__}')" 2>/dev/null || echo -e "    ${YELLOW}fastapi: NOT FOUND${NC}"
else
    echo -e "  ${RED}[X] Python not found!${NC}"
fi

# ------------------------------------------------------------
# 12. Jupyter config (optional)
# ------------------------------------------------------------
echo -e "\n${YELLOW}[12/12] Jupyter config:${NC}"
if [ -f ~/.jupyter/jupyter_server_config.py ]; then
    WEBSOCKET_SIZE=$(grep "websocket_max_message_size" ~/.jupyter/jupyter_server_config.py 2>/dev/null || echo "  Not configured")
    echo -e "  ${GREEN}[√] Jupyter config found: $WEBSOCKET_SIZE${NC}"
else
    echo -e "  ${YELLOW}[!] Jupyter config file not found (optional)${NC}"
fi

# ------------------------------------------------------------
# System load (bonus)
# ------------------------------------------------------------
echo -e "\n${CYAN}[+] System load average:${NC}"
uptime 2>/dev/null | awk '{print "  " $0}' || echo "  N/A"

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ System check completed!${NC}"
echo -e "${YELLOW}Tip: Run 'tail -f train.log' to monitor training progress${NC}"
echo -e "${BLUE}========================================${NC}"