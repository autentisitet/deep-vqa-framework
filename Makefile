GREEN  := \033[0;32m
BLUE   := \033[0;34m
RED    := \033[0;31m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
BOLD   := \033[1m
RESET  := \033[0m


PROJECT_NAME := Deep-VQA-Framework
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
LOG_DIR := $(ROOT_DIR)/results/scripts_logs
SECURITY_DIR := $(ROOT_DIR)/reports/security

$(shell mkdir -p $(LOG_DIR) $(SECURITY_DIR))


# ============================================================
# Python Detection (cross-platform)
# ============================================================
ifeq ($(OS),Windows_NT)
    PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/Scripts/python.exe" ]; then echo "$(ROOT_DIR)/.venv/Scripts/python.exe"; else echo "python"; fi)
else
    PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/bin/python" ]; then echo "$(ROOT_DIR)/.venv/bin/python"; else echo "python3"; fi)
endif
PYTHON := $(PYTHON_CMD)
UV_RUN := uv run



# ============================================================
# Targets
# ============================================================
.PHONY: help bootstrap setup install data info clean archive

.PHONY: test-images test-videos test-all

.PHONY: check-code fmt black isort format-all typecheck
.PHONY: vuln-audit sbom safety security-all

.PHONY: docker-dev docker-train docker-infer docker-stop docker-manage
.PHONY: docker-purge-all



.DEFAULT_GOAL := help



# ------------------------------------------------------------
# help - Show available commands
# ------------------------------------------------------------
help:
	@echo '$(BOLD)$(CYAN)Deep-VQA-Framework Makefile$(RESET)'
	@echo ''
	@echo '$(GREEN)Environment:$(RESET)'

	@echo '  make bootstrap [BOOTSTRAP_ARGS="..."]      Install system dependencies (apt)'
	@echo '  make setup [SETUP_ARGS="..."]              Install Python dependencies (uv)'
	@echo '  make install [INSTALL_ARGS="..."]          Bootstrap + Setup (full installation)'
	@echo '  make data                                  Download and prepare datasets'

	@echo ''
	@echo '$(YELLOW)Code Quality:$(RESET)'
	@echo '  make check-code     Run ruff linter'
	@echo '  make format-all     Format code (black + isort + ruff)'
	@echo '  make typecheck      Run mypy type checker'
	@echo ''
	@echo '$(RED)Security:$(RESET)'
	@echo '  make vuln-audit     Scan dependencies for CVEs'
	@echo '  make sbom           Generate CycloneDX SBOM'
	@echo '  make security-all   Run all security checks'
	@echo ''
	@echo '$(CYAN)Docker:$(RESET)'

	@echo '  make docker-dev [BUILD_ARGS="..."]         Enter development container'
	@echo '  make docker-train [BUILD_ARGS="..."]       Run training in background'
	@echo '  make docker-infer [BUILD_ARGS="..."]       Start inference API service'
	@echo '  make docker-stop                           Stop all containers'
	@echo '  make docker-manage                         Check container environment'
	@echo '  make docker-purge                          Remove all project containers/images'
	@echo ''
	@echo '$(BLUE)Inference:$(RESET)'
	@echo '  make test-images     Batch inference on examples/images/'
	@echo '  make test-videos     Batch inference on examples/videos/'
	@echo '  make test-all        Batch inference on all examples/'

	@echo ''
	@echo '$(BLUE)Maintenance:$(RESET)'
	@echo '  make clean          Remove cache and temporary files'
	@echo '  make archive        Package results'
	@echo ''
	@echo '$(CYAN)Info:$(RESET)'
	@echo '  make info           Show environment details'
	@echo ''

	@echo '$(BOLD)Parameters:$(RESET)'
	@echo '  BOOTSTRAP_ARGS="--mirror"    Pass args to bootstrap.sh'
	@echo '  SETUP_ARGS="--mirror --all"  Pass args to setup_env.sh'
	@echo '  INSTALL_ARGS="..."           Pass args to install (bootstrap + setup)'
	@echo '  BUILD_ARGS="--no-cache"      Pass args to docker build'
	@echo ''
	@echo '$(BOLD)Examples:$(RESET)'
	@echo '  make bootstrap BOOTSTRAP_ARGS="--mirror"'
	@echo '  make setup SETUP_ARGS="--mirror --all"'
	@echo '  make install INSTALL_ARGS="--mirror --all"'
	@echo '  make test-all'
	@echo '  uv run python -m src.main --dataset tid2013 --model swin_iqa'
	@echo '  uv run python -m src.main --dataset konvid-1k --model swin_vqa'



INSTALL_ARGS ?=
BOOTSTRAP_ARGS ?= $(filter --mirror, $(ARGS))
SETUP_ARGS ?= $(ARGS)
bootstrap:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@cd $(ROOT_DIR)/scripts && bash bootstrap.sh $(BOOTSTRAP_ARGS) 2>&1 | tee $(LOG_DIR)/bootstrap.log
	@echo "$(GREEN)[OK]$(RESET) Bootstrap complete."



setup:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@cd $(ROOT_DIR)/scripts && bash setup_env.sh $(SETUP_ARGS) 2>&1 | tee $(LOG_DIR)/setup_env.log
	@echo "$(GREEN)[OK]$(RESET) Setup complete. Run 'make info' to verify."



install: bootstrap setup
	@echo "$(GREEN)[OK]$(RESET) Full installation complete!"
	@echo "  System: bootstrap.sh"
	@echo "  Python: setup_env.sh"
	@echo "  Run 'make info' to verify environment."



data:
	@echo "$(BLUE)[INFO]$(RESET) Preparing datasets..."
	@cd $(ROOT_DIR)/scripts && bash manage_data.sh 2>&1 | tee $(LOG_DIR)/manage_data.log
	@echo "$(GREEN)[OK]$(RESET) Data ready."




clean:
	@echo "$(YELLOW)[WARN]$(RESET) Cleaning caches..."
	@read -p "Remove all caches? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		bash $(ROOT_DIR)/scripts/cache_clean.sh 2>&1 | tee $(LOG_DIR)/cache_clean.log; \
		echo "$(GREEN)[OK]$(RESET) Clean completed."; \
	else \
		echo "$(BLUE)[INFO]$(RESET) Clean aborted."; \
	fi




archive:
	@echo "$(BLUE)[INFO]$(RESET) Archiving results..."
	@if [ -f "$(LOG_DIR)/archive.log" ]; then \
		mv "$(LOG_DIR)/archive.log" "$(LOG_DIR)/archive.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@cd $(ROOT_DIR)/scripts && bash archive_results.sh --all 2>&1 | tee $(LOG_DIR)/archive.log
	@echo "$(GREEN)[OK]$(RESET) Archive completed."






# ============================================================
# Testing
# ============================================================

test-images:
	@echo "[INFO] Testing images..."
	@uv run python -m deploy.cli -i examples/images/
	@echo "[OK] Results saved under reports/iqa-test/"

test-videos:
	@echo "[INFO] Testing videos..."
	@uv run python -m deploy.cli -i examples/videos/
	@echo "[OK] Results saved under reports/vqa-test/"

test-all: test-images test-videos
	@echo "[OK] All tests completed"
	@jq -s '.[] | .[] | {file: .file, mos: .mos_score}' reports/iqa-test/*.json reports/vqa-test/*.json 2>/dev/null || echo "[WARN] jq not installed, check JSON files manually"





# ------------------------------------------------------------
# Code Quality
# ------------------------------------------------------------
check-code:
	@echo "$(YELLOW)[INFO]$(RESET) Running code quality checks..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show ruff >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) ruff not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run ruff format --check . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) All checks passed."

fmt:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting code with ruff..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show ruff >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) ruff not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run ruff format . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . --fix 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Formatting complete."

black:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting with black..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show black >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) black not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run black src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) black complete."

isort:
	@echo "$(YELLOW)[INFO]$(RESET) Sorting imports with isort..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show isort >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) isort not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run isort src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) isort complete."

format-all: black isort fmt
	@echo "$(GREEN)[OK]$(RESET) All formatters completed."

typecheck:
	@echo "$(YELLOW)[INFO]$(RESET) Running mypy type checks..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show mypy >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) mypy not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run mypy src/ --ignore-missing-imports 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Type checks complete."




# ------------------------------------------------------------
# Security
# ------------------------------------------------------------
vuln-audit:
	@echo "$(RED)[INFO]$(RESET) Auditing dependencies for vulnerabilities..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show pip-audit >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) pip-audit not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"'." && exit 1)
	@cd $(ROOT_DIR) && uv pip freeze > $(ROOT_DIR)/requirements.txt
	@cd $(ROOT_DIR) && uv run pip-audit \
		--requirement $(ROOT_DIR)/requirements.txt \
		--format json \
		--output $(SECURITY_DIR)/audit-report.json \
		--desc || true
	@cd $(ROOT_DIR) && uv run pip-audit \
		--requirement $(ROOT_DIR)/requirements.txt \
		--format columns \
		| tee $(SECURITY_DIR)/audit-report.txt
	@echo "$(GREEN)[OK]$(RESET) Audit complete. Reports: $(CYAN)$(SECURITY_DIR)/$(RESET)"

sbom:
	@echo "$(BLUE)[INFO]$(RESET) Generating SBOM (CycloneDX)..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show cyclonedx-bom >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) cyclonedx-bom not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"'." && exit 1)
	@cd $(ROOT_DIR) && uv run cyclonedx-py environment \
		--output-format json \
		--output-file $(SECURITY_DIR)/sbom-cyclonedx.json 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run cyclonedx-py environment \
		--output-format xml \
		--output-file $(SECURITY_DIR)/sbom-cyclonedx.xml 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv pip freeze > $(SECURITY_DIR)/dependencies.txt
	@cd $(ROOT_DIR) && uv pip tree > $(SECURITY_DIR)/dependency-tree.txt 2>/dev/null || true
	@echo "$(GREEN)[OK]$(RESET) SBOM generated. Reports: $(CYAN)$(SECURITY_DIR)/$(RESET)"

safety:
	@echo "$(YELLOW)[INFO]$(RESET) Running safety scan..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	@uv run pip show safety >/dev/null 2>&1 || (echo "$(RED)[ERROR]$(RESET) safety not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"'." && exit 1)
	@cd $(ROOT_DIR) && uv pip freeze > $(ROOT_DIR)/requirements.txt
	@cd $(ROOT_DIR) && uv run safety scan --full-report 2>&1 | tee $(SECURITY_DIR)/safety-report.txt || true
	@echo "$(GREEN)[OK]$(RESET) Safety scan complete. Report: $(CYAN)$(SECURITY_DIR)/safety-report.txt$(RESET)"

security-all: vuln-audit sbom safety
	@echo "$(GREEN)[OK]$(RESET) All security checks completed."
	@echo "Reports: $(CYAN)$(SECURITY_DIR)/$(RESET)"




# ------------------------------------------------------------
# info - Show environment and system details
# ------------------------------------------------------------
info:
	@echo "$(BOLD)$(CYAN)Deep-VQA-Framework - Environment Info$(RESET)"
	@echo "========================================================================"
	@echo ""
	@echo "$(BOLD)$(BLUE)Project:$(RESET)"
	@echo "  Name:  $(PROJECT_NAME)"
	@echo "  Root:  $(CYAN)$(ROOT_DIR)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Python:$(RESET)"
	@echo "  Interpreter: $(PYTHON)"
	@echo "  Version:     $$($(PYTHON) --version 2>/dev/null || echo '$(RED)not found$(RESET)')"
	@echo "  uv:          $$(cd $(ROOT_DIR) && $(HOME)/.local/bin/uv --version 2>/dev/null | cut -d' ' -f2 || echo '$(RED)not found$(RESET)')"
	@echo "  pip:         $$(cd $(ROOT_DIR) && uv run pip --version 2>/dev/null | cut -d' ' -f2 || echo '$(RED)not found$(RESET)')"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directories:$(RESET)"
	@echo "  Logs:       $(CYAN)$(LOG_DIR)$(RESET)"
	@echo "  Security:   $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo "  Cache:      $(CYAN)$$HOME/.cache/uv$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)System:$(RESET)"
	@echo "  OS:          $$(uname -s) $$(uname -r)"
	@echo "  Architecture: $$(uname -m)"
	@echo "  Hostname:    $$(hostname)"
	@echo "  User:        $$(whoami)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Hardware:$(RESET)"
	@echo "  CPU:         $$(nproc) cores"
	@echo "  Memory:      $$(free -h | awk '/^Mem:/ {print $$2}') total, $$(free -h | awk '/^Mem:/ {print $$4}') available"
	@echo "  Disk:        $$(df -h $(ROOT_DIR) | awk 'NR==2 {print $$4}') available"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "  GPU:         $$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; \
		echo "  GPU Memory:  $$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) total"; \
		echo "  Driver:      $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"; \
		CUDA_VER=$$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | sed 's/.*CUDA Version: *//' || echo 'N/A'); \
		echo "  CUDA:        $$CUDA_VER"; \
	else \
		echo "  GPU:         $(YELLOW)not detected$(RESET)"; \
	fi
	@echo ""
	@if [ -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(BOLD)$(GREEN)Virtual Environment:$(RESET)"; \
		echo "  Status:      $(GREEN)active$(RESET)"; \
		PKG_COUNT=$$(cd $(ROOT_DIR) && uv pip list 2>/dev/null | wc -l); \
		echo "  Packages:    $$PKG_COUNT installed"; \
		echo ""; \
		echo "$(BOLD)Key Packages:$(RESET)"; \
		echo -n "  PyTorch:     "; \
		cd $(ROOT_DIR) && uv run python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  CUDA:        "; \
		cd $(ROOT_DIR) && uv run python -c "import torch; print('$(GREEN)available$(RESET)' if torch.cuda.is_available() else '$(YELLOW)not available$(RESET)')" 2>/dev/null || echo "$(YELLOW)unknown$(RESET)"; \
		echo -n "  OpenCV:      "; \
		cd $(ROOT_DIR) && uv run python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  NumPy:       "; \
		cd $(ROOT_DIR) && uv run python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  Pandas:      "; \
		cd $(ROOT_DIR) && uv run python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  ruff:        "; \
		cd $(ROOT_DIR) && uv run ruff --version 2>/dev/null | head -1 | awk '{print $$2}' || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  black:       "; \
		cd $(ROOT_DIR) && uv run python -c "import black; print(black.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  isort:       "; \
		cd $(ROOT_DIR) && uv run python -c "import isort; print(isort.__version__)" 2>/dev/null || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  mypy:        "; \
		cd $(ROOT_DIR) && uv run mypy --version 2>/dev/null | head -1 | awk '{print $$2}' || echo "$(YELLOW)not installed$(RESET)"; \
		echo ""; \
		echo "$(BOLD)Security Tools:$(RESET)"; \
		echo -n "  pip-audit:   "; \
		cd $(ROOT_DIR) && uv run pip show pip-audit >/dev/null 2>&1 && echo "$(GREEN)installed$(RESET)" || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  cyclonedx:   "; \
		cd $(ROOT_DIR) && uv run pip show cyclonedx-bom >/dev/null 2>&1 && echo "$(GREEN)installed$(RESET)" || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  safety:      "; \
		cd $(ROOT_DIR) && uv run pip show safety >/dev/null 2>&1 && echo "$(GREEN)installed$(RESET)" || echo "$(YELLOW)not installed$(RESET)"; \
	else \
		echo "$(YELLOW)[WARN] Virtual environment not found. Run 'make setup' first.$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)Security Reports:$(RESET)"
	@if [ -d "$(SECURITY_DIR)" ]; then \
		FILE_COUNT=$$(find $(SECURITY_DIR) -type f 2>/dev/null | wc -l); \
		echo "  Directory:  $(CYAN)$(SECURITY_DIR)$(RESET)"; \
		echo "  Files:      $$FILE_COUNT"; \
		if [ -f "$(SECURITY_DIR)/audit-report.json" ]; then \
			SIZE=$$(ls -lh $(SECURITY_DIR)/audit-report.json 2>/dev/null | awk '{print $$5}'); \
			echo "  audit.json: $$SIZE"; \
		fi; \
		if [ -f "$(SECURITY_DIR)/sbom-cyclonedx.json" ]; then \
			SIZE=$$(ls -lh $(SECURITY_DIR)/sbom-cyclonedx.json 2>/dev/null | awk '{print $$5}'); \
			echo "  sbom.json:  $$SIZE"; \
		fi; \
		if [ -f "$(SECURITY_DIR)/safety-report.txt" ]; then \
			SIZE=$$(ls -lh $(SECURITY_DIR)/safety-report.txt 2>/dev/null | awk '{print $$5}'); \
			echo "  safety.txt: $$SIZE"; \
		fi; \
	else \
		echo "  $(YELLOW)No reports found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)Logs:$(RESET)"
	@if [ -d "$(LOG_DIR)" ]; then \
		FILE_COUNT=$$(find $(LOG_DIR) -type f 2>/dev/null | wc -l); \
		echo "  Directory: $(CYAN)$(LOG_DIR)$(RESET)"; \
		echo "  Files:     $$FILE_COUNT"; \
		echo "  Recent:"; \
		ls -lt $(LOG_DIR)/*.log 2>/dev/null | head -5 | awk '{printf "    %s (%s)\n", $$9, $$5}' || echo "    $(YELLOW)no logs found$(RESET)"; \
	else \
		echo "  $(YELLOW)Directory not found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(GREEN)Quick Commands:$(RESET)"
	@echo "  make setup              Install dependencies"
	@echo "  make data               Prepare datasets"
	@echo "  make typecheck          Run mypy type checker"
	@echo "  make format-all         Format all code"
	@echo "  make security-all       Run all security checks"
	@echo "  uv run python -m src.main --dataset tid2013 --model swin_iqa"
	@echo "  uv run python -m src.main --dataset konvid-1k --model swin_vqa"
	@echo "========================================================================"




# ============================================================
# Docker Commands
# ============================================================
COMPOSE := $(shell \
	if command -v podman-compose >/dev/null 2>&1; then \
		echo "podman-compose"; \
	elif command -v docker-compose >/dev/null 2>&1; then \
		echo "docker-compose"; \
	elif docker compose version >/dev/null 2>&1; then \
		echo "docker compose"; \
	else \
		echo ""; \
	fi)


RUNTIME := $(shell \
	if command -v podman >/dev/null 2>&1; then echo "podman"; \
	elif command -v docker >/dev/null 2>&1; then echo "docker"; \
	else echo ""; fi)


COMPOSE_FILES := -f docker/docker-compose.yaml
ifeq ($(RUNTIME),podman)
    COMPOSE_FILES += -f docker/docker-compose.podman.yaml
endif
ifeq ($(RUNTIME),docker)
    COMPOSE_FILES += -f docker/docker-compose.docker.yaml
endif


define check_runtime
	@if [ -z "$(RUNTIME)" ]; then \
		echo "ERROR: No container runtime found (podman or docker)"; \
		echo "Install: Fedora: sudo dnf install podman podman-compose"; \
		echo "        Ubuntu: sudo apt install docker.io docker-compose"; \
		exit 1; \
	fi
	@if [ -z "$(COMPOSE)" ]; then \
		echo "ERROR: No compose tool found"; \
		echo "Install: Fedora: sudo dnf install podman-compose"; \
		echo "        Ubuntu: sudo apt install docker-compose"; \
		exit 1; \
	fi
endef


BUILD_ARGS ?=


docker-dev:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-train
	$(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-train vqa-train


docker-train:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-train
	$(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-train vqa-train bash -c "\
		set -o pipefail && \
		make data && \
		uv run python -m src.main --dataset tid2013 --model swin_iqa 2>&1 | tee /app/results/tid2013.log && \
		uv run python -m src.main --dataset konvid-1k --model swin_vqa 2>&1 | tee /app/results/konvid-1k.log \
	"


docker-infer:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-infer
	$(COMPOSE) $(COMPOSE_FILES) up -d vqa-infer nginx



docker-stop:
	$(call check_runtime)
	@echo "[INFO] Stopping containers..."
	@$(RUNTIME) stop vqa-prod 2>/dev/null || true
	@$(RUNTIME) stop vqa-nginx 2>/dev/null || true
	@$(RUNTIME) stop vqa-train 2>/dev/null || true
	@echo "$(GREEN)[OK]$(RESET) Containers stopped."



docker-manage:
	$(call check_runtime)
	@$(RUNTIME) images --filter "dangling=true" -q | xargs -r $(RUNTIME) rmi -f
	@echo "Runtime: $(RUNTIME)"
	@echo "Compose: $(COMPOSE)"
	@echo "Files:   $(COMPOSE_FILES)"
	@echo ""
	@$(RUNTIME) --version 2>/dev/null || echo "ERROR: runtime not found"
	@echo ""
	@if [ "$(RUNTIME)" = "docker" ]; then \
		echo "--- Docker Service ---"; \
		sudo systemctl status docker 2>/dev/null | head -5 || echo "  (service status unavailable)"; \
		echo ""; \
	fi
	@echo "--- Container Stats ---"
	@$(RUNTIME) stats --no-stream 2>/dev/null || echo "  (no running containers)"
	@echo ""
	@echo "--- Images ---"
	@$(RUNTIME) images 2>/dev/null | head -10
	@echo ""
	@echo "--- Containers ---"
	@$(RUNTIME) ps -a 2>/dev/null | head -10
	@echo ""
	@echo "--- Volumes ---"
	@$(RUNTIME) volume ls 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "--- Networks ---"
	@$(RUNTIME) network ls 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "--- Compose Services ---"
	@$(COMPOSE) $(COMPOSE_FILES) ps 2>/dev/null || echo "  (none)"



docker-purge-all:
	$(call check_runtime)
	@echo "[INFO] Stopping and removing containers..."
	@$(COMPOSE) $(COMPOSE_FILES) down --remove-orphans 2>/dev/null || true
	@$(RUNTIME) ps -a --filter "name=vqa-train" -q | xargs -r $(RUNTIME) rm -f
	@$(RUNTIME) ps -a --filter "name=vqa-prod" -q | xargs -r $(RUNTIME) rm -f
	@$(RUNTIME) ps -a --filter "name=vqa-nginx" -q | xargs -r $(RUNTIME) rm -f

	@echo "[INFO] Removing dangling images..."
	@$(RUNTIME) images --filter "dangling=true" -q | xargs -r $(RUNTIME) rmi -f

	@echo "[INFO] Purging volumes..."
	@$(RUNTIME) volume prune -f 2>/dev/null || true

	@echo "[INFO] Purging networks..."
	@$(RUNTIME) network prune -f 2>/dev/null || true

	@echo "$(GREEN)[OK]$(RESET) Docker/Podman purge complete."
