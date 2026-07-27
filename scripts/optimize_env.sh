#!/usr/bin/env bash
# --- optimize_env.sh ---

# Tasks:
# 1. release ports
# 2. change network cards' mtu
# 3. check proxy; set http/https config if needed
# 4. Websocket control
#    - iopub_data_rate_limit
#    - rate_limit_window
#    - terminado_settings: inactive_timeout, ping_interval
#    - tornado_settings
#    - websocket_ping_interval
#    - websocket_ping_timeout
#    - iopub_msg_rate_limit
#    - allow_origin
#    - allow_remote_access
#    - disable_check_xsrf
# 5. Terminal silencing and flow control
# 6. Kernel idle cleanup
#    - cull_idle_timeout
#    - cull_connected
#    - cull_busy


# Notes:
# YAML/JSON separated "data-driven" programming is terrible in that case.
set -e -o pipefail

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}=== Network & Terminal Optimization ===${NC}"
echo -e "${BLUE}========================================${NC}"

# Helper: sudo handling
HAS_SUDO=false
if command -v sudo &> /dev/null; then
    HAS_SUDO=true
fi

if [ "$(id -u)" -eq 0 ]; then
    ADMIN_SUDO=""
else
    if [ "$HAS_SUDO" = true ]; then
        ADMIN_SUDO="sudo"
    else
        ADMIN_SUDO=""
        echo -e "${YELLOW}[!] No sudo available, some operations may fail${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# 1. Release ports (kill processes occupying common ports)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Releasing common ports...${NC}"
COMMON_PORTS=(6006 8888 8080 8000 7860)
for port in "${COMMON_PORTS[@]}"; do
    # Find PID using the port
    PID=$(ss -tlnp 2>/dev/null | grep -E ":$port " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [ -n "$PID" ]; then
        echo -e "  Port $port: killing process $PID"
        kill -9 "$PID" 2>/dev/null || true
    else
        echo -e "  Port $port: free"
    fi
done

# -----------------------------------------------------------------------------
# 2. Change network card's MTU
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Adjusting MTU settings...${NC}"
# Try common interface names: eth0, ens3, ens5, eno1
for iface in eth0 ens3 ens5 eno1; do
    if ip link show "$iface" &>/dev/null; then
        ${ADMIN_SUDO} ip link set dev "$iface" mtu 1400 2>/dev/null && \
            echo -e "  ${GREEN}[√] Set $iface MTU to 1400${NC}" || \
            echo -e "  ${YELLOW}[!] Failed to set MTU for $iface${NC}"
        break
    fi
done

# -----------------------------------------------------------------------------
# 3. Check proxy; set http/https config if needed
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Checking proxy configuration...${NC}"

if [ -n "$http_proxy" ] || [ -n "$HTTP_PROXY" ]; then
    echo -e "  ${GREEN}[√] Active environment proxy detected: ${http_proxy:-$HTTP_PROXY}${NC}"
elif curl -s -I --max-time 3 "https://www.google.com" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|301|302"; then
    echo -e "  ${GREEN}[√] Direct internet connection works, no local proxy needed${NC}"
else
    echo -e "  ${YELLOW}[!] Direct connection failed and no proxy set. Searching local proxy ports...${NC}"
    FOUND_PROXY=""
    for proxy_port in 7890 7897 10809 8118 12639; do
        if curl -s -I --max-time 2 --proxy "http://127.0.0.1:$proxy_port" \
            "https://www.google.com" -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|301|302"; then
            export http_proxy="http://127.0.0.1:$proxy_port"
            export https_proxy="$http_proxy"
            FOUND_PROXY="http://127.0.0.1:$proxy_port"
            echo -e "  ${GREEN}[√] Local proxy found & configured: $FOUND_PROXY${NC}"
            break
        fi
    done
    if [ -z "$FOUND_PROXY" ]; then
        echo -e "  ${YELLOW}[!] No local proxy detected. If using AutoDL, consider running 'source /etc/network_turbo'${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# 4. Jupyter/WebSocket optimization (full configuration with compatibility)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Configuring Jupyter WebSocket settings...${NC}"
mkdir -p ~/.jupyter

# Write dual-compatible Jupyter configurations
python3 -c '
import os

jupyter_dir = os.path.expanduser("~/.jupyter")

# Config block for jupyter_server_config.py (JupyterLab / Server v2+)
server_block = """
# ========== Network & WebSocket Optimization for Jupyter Server ==========
c.ServerApp.tornado_settings = {
    "websocket_max_message_size": 500 * 1024 * 1024,   # 500MB
    "websocket_ping_interval": 30000,                  # 30s
    "websocket_ping_timeout": 30000,                   # 30s
}
c.ServerApp.iopub_data_rate_limit = 10000000           # 10MB/s
c.ServerApp.rate_limit_window = 3.0
c.ServerApp.iopub_msg_rate_limit = 5000
c.ServerApp.terminado_settings = {
    "inactive_timeout": 600,
    "ping_interval": 60,
}
c.ServerApp.allow_origin = "*"
c.ServerApp.allow_remote_access = True
"""

# Config block for jupyter_notebook_config.py (Classic Notebook v6)
notebook_block = """
# ========== Network & WebSocket Optimization for Classic Notebook ==========
c.NotebookApp.tornado_settings = {
    "websocket_max_message_size": 500 * 1024 * 1024,
    "websocket_ping_interval": 30000,
    "websocket_ping_timeout": 30000,
}
c.NotebookApp.iopub_data_rate_limit = 10000000
c.NotebookApp.rate_limit_window = 3.0
c.NotebookApp.iopub_msg_rate_limit = 5000
c.NotebookApp.allow_origin = "*"
c.NotebookApp.allow_remote_access = True
"""

kernel_cull_block = """
# ========== Kernel Idle Cleanup Policy ==========
c.MappingKernelManager.cull_idle_timeout = 86400       # 24 hours idle timeout
c.MappingKernelManager.cull_connected = False          # Don"t cull connected kernels
c.MappingKernelManager.cull_busy = False               # Don"t cull busy kernels
"""

def apply_config(filename, main_block):
    path = os.path.join(jupyter_dir, filename)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# Generated by Network Control Script\n")

    content = open(path).read()
    if "websocket_max_message_size" not in content:
        with open(path, "a") as f:
            f.write(main_block)
            f.write(kernel_cull_block)
        print(f"  [√] Applied to {filename}")
    else:
        print(f"  [i] {filename} already optimized")

apply_config("jupyter_server_config.py", server_block)
apply_config("jupyter_notebook_config.py", notebook_block)
'

# -----------------------------------------------------------------------------
# 5. Terminal silencing and flow control
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Configuring terminal settings...${NC}"
BASHRC="$HOME/.bashrc"

# Disable Ctrl+S flow control and set colorful prompt
if ! grep -q "stty -ixon" "$BASHRC" 2>/dev/null; then
    cat <<'EOF' >> "$BASHRC"

# ========== Terminal Optimization for AutoDL ==========
# Disable Ctrl+S flow control (prevents accidental terminal freeze)
stty -ixon 2>/dev/null

# Colorful prompt: green user@host, blue path
export PS1="\[\e[32m\]\u@autodl\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ "

# Quick switch agent
alias turbo_on='source /etc/network_turbo 2>/dev/null || echo "network_turbo not found"'
alias proxy_on='export http_proxy=http://127.0.0.1:7890; export https_proxy=http://127.0.0.1:7890; echo "Proxy ON (127.0.0.1:7890)"'
alias proxy_off='unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; echo "Proxy OFF"'

# High performance aria2 alias (uses current environment proxy dynamically)
alias aria2p='aria2c --check-certificate=false -x 16 -s 16 -k 1M'

ulimit -n 65535
EOF
    echo -e "  ${GREEN}[√] Terminal settings added to ~/.bashrc${NC}"
else
    echo -e "  [i] Terminal settings already configured"
fi

# Apply settings to current session (if running interactively)
if [ -t 0 ]; then
    stty -ixon 2>/dev/null || true
    export PS1="\[\e[32m\]\u@autodl\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ "
fi

# -----------------------------------------------------------------------------
# 6. Kernel idle cleanup (already done in Jupyter config, just a reminder)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Kernel idle cleanup configuration...${NC}"
echo -e "  ${GREEN}[√] Idle timeout set to 24 hours (cull_idle_timeout=86400)${NC}"
echo -e "  ${GREEN}[√] Connected/Busy kernels will NOT be culled${NC}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Network & Terminal optimization complete!${NC}"
echo -e "Note: 'iopub_data_rate_limit' has no effect on kernels that have already started.${NC}"
echo -e "Note: 'WebSocket size limit': The system must be shut down and then restarted (the instance must be restarted).${NC}"
echo -e "${YELLOW}Note: Please run 'source ~/.bashrc' and restart the kernel to apply prompt changes${NC}"
echo -e "${YELLOW}Note: Restart the kernel may quit from uv venv.${NC}"
echo -e "${BLUE}========================================${NC}"