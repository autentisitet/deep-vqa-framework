ESC    := $(shell printf '\033')
GREEN  := $(ESC)[0;32m
BLUE   := $(ESC)[0;34m
RED    := $(ESC)[0;31m
YELLOW := $(ESC)[0;33m
CYAN   := $(ESC)[0;36m
BOLD   := $(ESC)[1m
RESET  := $(ESC)[0m

SHELL := /bin/bash


PROJECT_NAME := Deep-VQA-Framework
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
LOG_DIR := $(ROOT_DIR)/results/scripts_logs
SECURITY_DIR := $(ROOT_DIR)/reports/security

$(shell mkdir -p $(LOG_DIR) $(SECURITY_DIR))


# ============================================================
# Python Detection (cross-platform)
# ============================================================
ifeq ($(OS),Windows_NT)
    PYTHON_CMD := $(shell \
        if [ -f "$(ROOT_DIR)/.venv/Scripts/python.exe" ]; then \
            echo "$(ROOT_DIR)/.venv/Scripts/python.exe"; \
        else echo "python"; fi)
else
    PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/bin/python" ]; then echo "$(ROOT_DIR)/.venv/bin/python"; else echo "python3"; fi)
endif
PYTHON := $(PYTHON_CMD)
UV_RUN := uv run



# ============================================================
# Targets
# ============================================================
.PHONY: help help-% bootstrap setup install env-secrets data info cache_clean archive results-clean

.PHONY: git-log version-sync version-check

.PHONY: pytest test-images test-videos test-all

.PHONY: check-code fmt black isort format-all typecheck
.PHONY: vuln-audit sbom safety security-all

.PHONY: docker-dev docker-train docker-stop docker-manage
.PHONY: docker-purge-all
.PHONY: docker-infer docker-infer-internal docker-infer-public
.PHONY: docker-infer-internal-ollama docker-infer-public-ollama docker-deploy-check



.DEFAULT_GOAL := help



# ------------------------------------------------------------
# help - Show available commands
# ------------------------------------------------------------
help:
	@echo '$(BOLD)$(CYAN)Deep-VQA-Framework Makefile$(RESET)'
	@echo ''
	@echo '$(GREEN)Environment:$(RESET)'

	@echo '  make bootstrap [MIRROR=1]                   Install system dependencies (apt)'
	@echo '  make setup [DEV=1 SECURITY=1 MIRROR=1]      Install Python dependencies (uv)'
	@echo '  make install [DEV=1 SECURITY=1 MIRROR=1]    Bootstrap + Setup (full installation)'
	@echo '  make env-secrets                            Create .env and generate API/cookie secrets'
	@echo '  make data                                  Download and prepare datasets'

	@echo ''
	@echo '$(YELLOW)Code Quality:$(RESET)'
	@echo '  make check-code     Run ruff linter'
	@echo '  make format-all     Format code (black + isort + ruff)'
	@echo '  make typecheck      Run mypy type checker'
	@echo '  make version-check  Verify README/runtime version markers match pyproject.toml'
	@echo ''
	@echo '$(RED)Security:$(RESET)'
	@echo '  make vuln-audit     Scan dependencies for CVEs'
	@echo '  make sbom           Generate CycloneDX SBOM'
	@echo '  make security-all   Run all security checks'
	@echo ''
	@echo '$(CYAN)Docker:$(RESET)'

	@echo '  make docker-dev [BUILD_ARGS="..."]         Open a dev shell (dev tools included)'
	@echo '  make docker-train [DEV=1 BUILD_ARGS="..."] Run the configured training jobs'
	@echo '  make docker-stop                           Stop project containers, including Ollama if running'
	@echo '  make docker-manage                         Inspect runtime and project resources'
	@echo '  make docker-purge-all                      Remove all project containers/images'
	@echo '  make docker-infer [DEPLOYMENT_MODE=public] Default inference stack'
	@echo '  make docker-infer-internal-ollama          Internal profile + Ollama'
	@echo '  make docker-infer-public-ollama            Public profile + Ollama'
	@echo '  make docker-deploy-check                   Check API and optional Nginx/Ollama services'
	@echo '  Note: inference commands create .env from .env.example when missing; internal mode still needs keys.'
	@echo ''
	@echo '$(BLUE)Inference:$(RESET)'
	@echo '  make test-images     Batch inference on examples/images/'
	@echo '  make test-videos     Batch inference on examples/videos/'
	@echo '  make test-all        Batch inference on all examples/'
	@echo '  make pytest          Run the tests/ pytest suite'

	@echo ''
	@echo '$(BLUE)Maintenance:$(RESET)'
	@echo '  make cache_clean    Remove cache and temporary files'
	@echo '  make results-clean       Delete dated .pt/.csv/.log files older than 3 days'
	@echo '  make archive [ARCHIVE_ARGS="..."]       Package results'
	@echo '  make git-log        Export git history and update ignore files'
	@echo ''
	@echo '$(CYAN)Info:$(RESET)'
	@echo '  make info           Show environment details'
	@echo ''

	@echo '$(BOLD)Parameters:$(RESET)'
	@echo '  DEV=1                       Include development tools'
	@echo '  SECURITY=1                  Include security tools'
	@echo '  MIRROR=1                    Use the TUNA package mirror'
	@echo '  DEPLOYMENT_MODE=internal|public  Select inference profile (default: internal)'
	@echo '  BUILD_ARGS="..."             Advanced Docker build flags'
	@echo '  ARCHIVE_ARGS="..."           Advanced archive script arguments'
	@echo '  GIT_LOG_FILE="..."            Git history output path'
	@echo '  BOOTSTRAP_ARGS/SETUP_ARGS    Advanced raw script arguments (optional)'
	@echo ''
	@echo '$(BOLD)Examples:$(RESET)'
	@echo '  make bootstrap MIRROR=1'
	@echo '  make setup DEV=1 MIRROR=1'
	@echo '  make install DEV=1 SECURITY=1 MIRROR=1'
	@echo '  make env-secrets'
	@echo '  make docker-infer'
	@echo '  make docker-infer DEPLOYMENT_MODE=public'
	@echo '  make docker-infer-internal-ollama'
	@echo '  make docker-train DEV=1'
	@echo '  make docker-train BUILD_ARGS="--build-arg USE_BUILD_PROXY=true"'
	@echo '  make archive ARCHIVE_ARGS="--results"'
	@echo '  make git-log GIT_LOG_FILE="results/git_log.txt"'
	@echo '  make test-all'
	@echo '  uv run python -m src.main --dataset tid2013 --model swin_iqa'
	@echo '  uv run python -m src.main --dataset konvid-1k --model swin_vqa'
	@echo ''
	@echo '  Detailed help: make help-TARGET (for example: make help-docker-infer)'
	@echo '  Knowledge base: knowledge/README.md'

help-%:
	@case "$*" in \
		setup) \
			echo 'COMMAND: make setup DEV=1 MIRROR=1'; \
			echo 'PURPOSE: Install Python/runtime dependencies with uv.' ;; \
		env-secrets) \
			echo 'COMMAND: make env-secrets'; \
			echo 'PURPOSE: Create .env and generate only missing deployment secrets.'; \
			echo 'NOTE: Existing DEEP_VQA_API_KEY and DEEP_VQA_AUTH_SECRET values are preserved.' ;; \
		data) \
			echo 'COMMAND: make data'; \
			echo 'PURPOSE: Prepare datasets through the data workflow.' ;; \
		 docker-dev) \
			echo 'COMMAND: make docker-dev [BUILD_ARGS="..."]'; \
			echo 'PURPOSE: Open an interactive container with development tools.'; \
			echo 'NOTE: tests/, source, configs, and caches are mounted from the host.' ;; \
		docker-train) \
			echo 'COMMAND: make docker-train [DEV=1] [BUILD_ARGS="..."]'; \
			echo 'PURPOSE: Run the configured training workflows.'; \
			echo 'NOTE: DEV=1 adds pytest/ruff/mypy/black/isort to the image.' ;; \
		docker-infer) \
			echo 'COMMAND: make docker-infer [DEPLOYMENT_MODE=internal|public]'; \
			echo 'PURPOSE: Start vqa-infer and vqa-nginx; default profile is internal.'; \
			echo 'NOTE: use make docker-infer-internal-ollama for subjective quality.' ;; \
		docker-infer-internal-ollama|docker-infer-public-ollama) \
			echo 'COMMAND: make '$*; \
			echo 'PURPOSE: Start vqa-ollama, initialize the model, then start inference.'; \
			echo 'NOTE: internal mode requires DEEP_VQA_API_KEY and DEEP_VQA_AUTH_SECRET.' ;; \
		docker-stop) \
			echo 'COMMAND: make docker-stop'; \
			echo 'PURPOSE: Stop project services without deleting data.' ;; \
		docker-purge-all) \
			echo 'COMMAND: make docker-purge-all'; \
			echo 'PURPOSE: Remove project containers, images, and volumes.' ;; \
		docker-deploy-check) \
			echo 'COMMAND: make docker-deploy-check'; \
			echo 'PURPOSE: Check API and optional Nginx/Ollama services.' ;; \
		test-images|test-videos|test-all) echo 'make $*'; echo '  Run deployment batch inference against the example media set.' ;; \
		pytest) echo 'make pytest'; echo '  Run the tests/ pytest suite.' ;; \
		*) echo "No detailed help is available for '$*'. Run 'make help' to list targets."; exit 1 ;; \
		esac

version-sync:
	@$(PYTHON) scripts/sync_version.py

version-check:
	@$(PYTHON) scripts/sync_version.py --check



DEV ?= 0
SECURITY ?= 0
MIRROR ?= 0
INSTALL_ARGS ?=
BOOTSTRAP_ARGS ?= $(if $(filter 1 true yes,$(MIRROR)),--mirror) $(filter --mirror,$(INSTALL_ARGS))
SETUP_ARGS ?= $(if $(filter 1 true yes,$(MIRROR)),--mirror) $(if $(filter 1 true yes,$(DEV)),--dev) $(if $(filter 1 true yes,$(SECURITY)),--security) $(INSTALL_ARGS)
ARCHIVE_ARGS ?= --all
GIT_LOG_FILE ?= $(ROOT_DIR)/git_log.txt
bootstrap:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@cd $(ROOT_DIR)/scripts && bash bootstrap.sh $(BOOTSTRAP_ARGS) 2>&1 | tee $(LOG_DIR)/bootstrap.log
	@echo "$(GREEN)[OK]$(RESET) Bootstrap complete."



setup:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@cd $(ROOT_DIR)/scripts && bash setup_env.sh $(SETUP_ARGS) 2>&1 | tee $(LOG_DIR)/setup_env.log
	@echo "$(GREEN)[OK]$(RESET) Setup complete. Run 'make info' to verify."


env-secrets:
	@chmod +x $(ROOT_DIR)/scripts/init_env_secrets.sh
	@bash $(ROOT_DIR)/scripts/init_env_secrets.sh



install: bootstrap setup
	@echo "$(GREEN)[OK]$(RESET) Full installation complete!"
	@echo "  System: bootstrap.sh"
	@echo "  Python: setup_env.sh"
	@echo "  Run 'make info' to verify environment."



data:
	@echo "$(BLUE)[INFO]$(RESET) Preparing datasets..."
	@cd $(ROOT_DIR)/scripts && bash manage_data.sh 2>&1 | tee $(LOG_DIR)/manage_data.log
	@echo "$(GREEN)[OK]$(RESET) Data ready."




cache_clean:
	@echo "$(YELLOW)[WARN]$(RESET) Cleaning caches..."
	@read -p "Remove all caches? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		bash $(ROOT_DIR)/scripts/cache_clean.sh 2>&1 | tee $(LOG_DIR)/cache_clean.log; \
		echo "$(GREEN)[OK]$(RESET) Clean completed."; \
	else \
		echo "$(BLUE)[INFO]$(RESET) Clean aborted."; \
	fi




results-clean:
	@set -e; \
	retention_before="$$(date -d '3 days ago' '+%Y%m%d_%H%M%S')"; \
	read -r -p "Delete dated .pt/.csv/.log files in results older than $$retention_before? [y/N] " confirm; \
	if [ "$$confirm" != "y" ]; then \
		echo "$(BLUE)[INFO]$(RESET) Results cleanup aborted."; \
		exit 0; \
	fi; \
	: > "$(LOG_DIR)/results_clean.log"; \
	count=0; \
	while IFS= read -r -d '' artifact; do \
		filename="$$(basename "$$artifact")"; \
		run_at="$${filename:0:15}"; \
		if [[ "$$run_at" =~ ^[0-9]{8}_[0-9]{6}$$ ]] && [[ "$$run_at" < "$$retention_before" ]]; then \
			printf "$(YELLOW)[INFO]$(RESET) Deleting artifact from %s: %s\n" "$$run_at" "$$filename"; \
			rm -f -- "$$artifact"; \
			printf '%s\n' "$$artifact" | tee -a "$(LOG_DIR)/results_clean.log"; \
			count=$$((count + 1)); \
		fi; \
	done < <(find "$(ROOT_DIR)/results" -type f \( -name '*.pt' -o -name '*.csv' -o -name '*.log' \) -print0); \
	echo "$(GREEN)[OK]$(RESET) Deleted $$count dated result file(s) older than 3 days. Log: $(CYAN)$(LOG_DIR)/results_clean.log$(RESET)"


archive:
	@echo "$(BLUE)[INFO]$(RESET) Archiving results..."
	@if [ -f "$(LOG_DIR)/archive.log" ]; then \
		mv "$(LOG_DIR)/archive.log" "$(LOG_DIR)/archive.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@cd $(ROOT_DIR)/scripts && bash archive_results.sh $(ARCHIVE_ARGS) 2>&1 | tee $(LOG_DIR)/archive.log
	@echo "$(GREEN)[OK]$(RESET) Archive completed."

git-log:
	@mkdir -p "$$(dirname "$(GIT_LOG_FILE)")"
	@git --no-pager log --oneline --graph --all --decorate > "$(GIT_LOG_FILE)"
	@ignore_path="$(GIT_LOG_FILE)"; \
	case "$$ignore_path" in \
		"$(ROOT_DIR)"/*) ignore_path="$${ignore_path#$(ROOT_DIR)/}" ;; \
		./*) ignore_path="$${ignore_path#./}" ;; \
	esac; \
	for ignore_file in "$(ROOT_DIR)/.gitignore" "$(ROOT_DIR)/.dockerignore"; do \
		if ! grep -Fxq -- "$$ignore_path" "$$ignore_file"; then \
			if [ -s "$$ignore_file" ] && [ "$$(tail -c 1 "$$ignore_file" | wc -l)" -eq 0 ]; then \
				printf '\n' >> "$$ignore_file"; \
			fi; \
			printf '%s\n' "$$ignore_path" >> "$$ignore_file"; \
			echo "$(GREEN)[OK]$(RESET) Added $$ignore_path to $$(basename "$$ignore_file")"; \
		fi; \
	done
	@echo "$(GREEN)[OK]$(RESET) Git history written to $(GIT_LOG_FILE)"






# ============================================================
# Testing
# ============================================================

define require_tool
	@uv pip show --python "$(PYTHON)" $(1) >/dev/null 2>&1 || { \
		echo "$(RED)[ERROR]$(RESET) $(1) not installed."; \
		echo "Run: make setup $(shell echo $(2) | tr a-z A-Z)=1 MIRROR=1"; \
		exit 1; \
	}
endef

pytest:
	@echo "$(BLUE)[INFO]$(RESET) Running tests/ with pytest..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,pytest,dev)
	@set -o pipefail; cd $(ROOT_DIR) && PYTHONPATH="$(ROOT_DIR)" uv run --no-sync pytest -q 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Pytest suite passed."

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
	@jq -s '.[] | .[] | {file: .file, mos: .mos_score}' \
		reports/iqa-test/*.json reports/vqa-test/*.json 2>/dev/null \
		|| echo "[WARN] jq not installed, check JSON files manually"





# ------------------------------------------------------------
# Code Quality
# ------------------------------------------------------------

check-code:
	@echo "$(YELLOW)[INFO]$(RESET) Running code quality checks..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,ruff,dev)
	@cd $(ROOT_DIR) && uv run ruff format --check . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) All checks passed."

fmt:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting code with ruff..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,ruff,dev)
	@cd $(ROOT_DIR) && uv run ruff format . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . --fix 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Formatting complete."

black:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting with black..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,black,dev)
	@cd $(ROOT_DIR) && uv run black src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) black complete."

isort:
	@echo "$(YELLOW)[INFO]$(RESET) Sorting imports with isort..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,isort,dev)
	@cd $(ROOT_DIR) && uv run isort src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) isort complete."

format-all: black isort fmt
	@echo "$(GREEN)[OK]$(RESET) All formatters completed."

typecheck:
	@echo "$(YELLOW)[INFO]$(RESET) Running mypy type checks..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,mypy,dev)
	@cd $(ROOT_DIR) && uv run mypy src/ --ignore-missing-imports 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Type checks complete."




# ------------------------------------------------------------
# Security
# ------------------------------------------------------------
vuln-audit:
	@echo "$(RED)[INFO]$(RESET) Auditing dependencies for vulnerabilities..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,pip-audit,security)
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
	$(call require_tool,cyclonedx-bom,security)
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
	$(call require_tool,safety,security)
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
		cd $(ROOT_DIR) && uv run python -c \
			"import torch; print('$(GREEN)available$(RESET)' if torch.cuda.is_available() else '$(YELLOW)not available$(RESET)')" \
			2>/dev/null || echo "$(YELLOW)unknown$(RESET)"; \
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
		cd $(ROOT_DIR) && uv run pip show pip-audit >/dev/null 2>&1 \
			&& echo "$(GREEN)installed$(RESET)" || echo "$(YELLOW)not installed$(RESET)"; \
		echo -n "  cyclonedx:   "; \
		cd $(ROOT_DIR) && uv run pip show cyclonedx-bom >/dev/null 2>&1 \
			&& echo "$(GREEN)installed$(RESET)" || echo "$(YELLOW)not installed$(RESET)"; \
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


COMPOSE_FILES := -f deploy-config/compose/docker-compose.yaml
ifeq ($(RUNTIME),podman)
    COMPOSE_FILES += -f deploy-config/compose/docker-compose.podman.yaml
endif
ifeq ($(RUNTIME),docker)
    COMPOSE_FILES += -f deploy-config/compose/docker-compose.docker.yaml
endif

# This must be expanded after all runtime-specific compose overrides.
OLLAMA_COMPOSE_FILES := $(COMPOSE_FILES) -f deploy-config/ollama/docker-compose.yaml
COMPOSE_ENV = DEEP_VQA_DEPLOYMENT_CONFIG="$(DEPLOYMENT_CONFIG)"
OLLAMA_COMPOSE_ENV = $(COMPOSE_ENV) OLLAMA_BASE_URL="$(OLLAMA_BASE_URL)" OLLAMA_MODELFILE_PATH="$(ROOT_DIR)/deploy-config/ollama/Modelfile"


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

define ensure_env_file
	@if [ ! -f "$(ROOT_DIR)/.env" ]; then \
		if [ ! -f "$(ROOT_DIR)/.env.example" ]; then \
			echo "ERROR: .env.example is missing."; exit 1; \
		fi; \
		cp "$(ROOT_DIR)/.env.example" "$(ROOT_DIR)/.env"; \
		echo "[INFO] Created .env from .env.example."; \
		echo "[INFO] Internal profile still requires DEEP_VQA_API_KEY and DEEP_VQA_AUTH_SECRET."; \
	fi
endef

define show_deployment_context
	@echo "$(CYAN)[INFO]$(RESET) Deployment mode: $(1)"
	@echo "$(CYAN)[INFO]$(RESET) Profile: $(2)"
	@echo "$(CYAN)[INFO]$(RESET) Auth: $(3)"
	@echo "$(CYAN)[INFO]$(RESET) Evaluation store: $(4)"
	@echo "$(CYAN)[INFO]$(RESET) Ollama endpoint (when enabled): $(OLLAMA_BASE_URL)"
endef


define check_deploy_auth
	@if grep -Eq '^[[:space:]]*mode:[[:space:]]*api_key([[:space:]]|$$)' $(DEPLOYMENT_CONFIG); then \
		api_key="$${DEEP_VQA_API_KEY:-}"; auth_secret="$${DEEP_VQA_AUTH_SECRET:-}"; \
		if [ -f .env ]; then \
			[ -n "$$api_key" ] || api_key=$$(sed -n 's/^DEEP_VQA_API_KEY=//p' .env | tail -n 1); \
			[ -n "$$auth_secret" ] || auth_secret=$$(sed -n 's/^DEEP_VQA_AUTH_SECRET=//p' .env | tail -n 1); \
		fi; \
		if [ -z "$$api_key" ] || [ -z "$$auth_secret" ]; then \
			echo "ERROR: auth.mode=api_key requires DEEP_VQA_API_KEY and DEEP_VQA_AUTH_SECRET."; \
			echo "Create local config first: cp .env.example .env"; \
			echo "Generate values with: openssl rand -hex 32"; \
			exit 1; \
		fi; \
	fi
endef

# Wait only while a container can still become ready. Exited/dead/unhealthy
# states fail immediately instead of consuming the entire timeout budget.
define wait_container_http
	@attempts=$(3); unconfigured_attempts=0; \
	for i in $$(seq 1 $$attempts); do \
		state=$$($(RUNTIME) inspect --format '{{.State.Status}}' $(1) \
			2>/dev/null || echo missing); \
		health=$$($(RUNTIME) inspect \
			--format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unconfigured{{end}}' \
			$(1) 2>/dev/null || echo missing); \
		case "$$state/$$health" in \
			missing/*) \
				echo "$(RED)[ERROR]$(RESET) $(1) is not present."; \
				exit 1 ;; \
			exited/*|dead/*|removing/*|*/unhealthy) \
				echo "$(RED)[ERROR]$(RESET) $(1) cannot become ready: state=$$state health=$$health"; \
				exit 1 ;; \
			running/healthy) \
				curl --noproxy '*' -fsS --max-time 5 "$(2)" >/dev/null 2>&1 \
					&& { echo "$(GREEN)[OK]$(RESET) $(1) is healthy."; exit 0; }; \
				echo "$(RED)[ERROR]$(RESET) $(1) health endpoint failed."; \
				exit 1 ;; \
			running/unconfigured) \
				unconfigured_attempts=$$((unconfigured_attempts + 1)) ;; \
			created/*|running/starting) ;; \
			*) unconfigured_attempts=$$((unconfigured_attempts + 1)) ;; \
		esac; \
		if curl --noproxy '*' -fsS --max-time 5 "$(2)" >/dev/null 2>&1; then \
			echo "$(GREEN)[OK]$(RESET) $(1) is healthy."; \
			exit 0; \
		fi; \
		if [ "$$unconfigured_attempts" -ge 2 ]; then \
			echo "$(RED)[ERROR]$(RESET) $(1) endpoint failed after two probes."; \
			exit 1; \
		fi; \
		[ "$$i" -lt "$$attempts" ] && sleep 2; \
	done; \
	echo "$(RED)[ERROR]$(RESET) $(1) did not become healthy."; \
	exit 1
endef


BUILD_ARGS ?=
DEPLOYMENT_MODE ?= internal
DEPLOYMENT_CONFIG ?= deploy-config/profiles/infer_deploy.$(DEPLOYMENT_MODE).yaml
OLLAMA_BASE_URL ?= http://vqa-ollama:11434
API_HEALTH_URL = http://127.0.0.1:$${API_PORT:-8001}/v1/health
WEB_HEALTH_URL = http://127.0.0.1:$${WEB_PORT:-8000}/api/v1/health
OLLAMA_HEALTH_URL = http://127.0.0.1:$${OLLAMA_PORT:-11434}/api/tags


docker-dev:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) --build-arg INSTALL_DEV=true vqa-dev
	$(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-dev vqa-dev


docker-train:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) --build-arg INSTALL_DEV=$(if $(filter 1 true yes,$(DEV)),true,false) vqa-train
	$(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-train vqa-train bash -c "\
		set -o pipefail && \
		make data && \
		uv run python -m src.main --dataset tid2013 --model swin_iqa 2>&1 | tee /app/results/tid2013.log && \
		uv run python -m src.main --dataset konvid-1k --model swin_vqa 2>&1 | tee /app/results/konvid-1k.log \
	"


docker-api:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-infer
	$(COMPOSE) $(COMPOSE_FILES) up -d vqa-infer
	@echo "[INFO] Waiting for FastAPI health at http://127.0.0.1:$${API_PORT:-8001}/v1/health..."
	@for i in $$(seq 1 40); do \
		if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$${API_PORT:-8001}/v1/health" >/dev/null 2>&1; then \
			echo "$(GREEN)[OK]$(RESET) FastAPI is healthy."; exit 0; \
		fi; sleep 3; \
	done; echo "$(RED)[ERROR]$(RESET) FastAPI did not become healthy."; $(COMPOSE) $(COMPOSE_FILES) logs --tail=80 vqa-infer; exit 1


docker-infer: docker-api
	$(COMPOSE) $(COMPOSE_FILES) up -d nginx
	@echo "[INFO] Verifying proxied API..."
	@curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$${WEB_PORT:-8000}/api/v1/health" || \
		(echo "$(RED)[ERROR]$(RESET) Nginx/API health check failed."; exit 1)
	@echo "$(GREEN)[OK]$(RESET) Inference stack is ready: http://127.0.0.1:$${WEB_PORT:-8000}/"



docker-stop:
	$(call check_runtime)
	@echo "[INFO] Stopping containers..."
	@$(RUNTIME) stop vqa-infer 2>/dev/null || true
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
	@$(OLLAMA_COMPOSE_ENV) $(COMPOSE) $(OLLAMA_COMPOSE_FILES) ps 2>/dev/null || echo "  (none)"



docker-purge-all:
	$(call check_runtime)
	@echo "[INFO] Stopping and removing containers..."
	@$(COMPOSE) $(COMPOSE_FILES) down --remove-orphans 2>/dev/null || true
	@$(RUNTIME) ps -a --filter "name=vqa-train" -q | xargs -r $(RUNTIME) rm -f
	@$(RUNTIME) ps -a --filter "name=vqa-infer" -q | xargs -r $(RUNTIME) rm -f
	@$(RUNTIME) ps -a --filter "name=vqa-nginx" -q | xargs -r $(RUNTIME) rm -f

	@echo "[INFO] Removing dangling images..."
	@$(RUNTIME) images --filter "dangling=true" -q | xargs -r $(RUNTIME) rmi -f

	@echo "[INFO] Purging volumes..."
	@$(RUNTIME) volume prune -f 2>/dev/null || true

	@echo "[INFO] Purging networks..."
	@$(RUNTIME) network prune -f 2>/dev/null || true

	@echo "$(GREEN)[OK]$(RESET) Docker/Podman purge complete."
