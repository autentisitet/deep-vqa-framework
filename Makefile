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

$(shell mkdir -p $(LOG_DIR))


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
.PHONY: help h '?' help-% bootstrap setup install data info cache-clean archive results-clean

.PHONY: git-log version-sync version-check

.PHONY: pytest docker-batch-test
.PHONY: _docker-batch-images _docker-batch-videos

.PHONY: check-code fmt black isort format-all typecheck

.PHONY: docker-dev docker-train docker-stop docker-manage local
.PHONY: docker-purge-train docker-purge-deploy docker-purge-all

.PHONY: env-secrets docker-infer account-disable account-enable account-reset-password
.PHONY: docker-deploy-check docker-memory-measure




.DEFAULT_GOAL := help



# ------------------------------------------------------------
# help - Show available commands
# ------------------------------------------------------------
help:
	@echo '$(BOLD)$(CYAN)Deep-VQA-Framework Makefile$(RESET)'
	@echo 'Project targets are listed below.'
	@echo 'Use `make`, `make help`, `make h`, or `make ?` for this index.'
	@echo '`make --help` shows GNU Make options, not project targets.'
	@echo ''
	@echo '$(GREEN)Environment:$(RESET)'

	@echo '  $(CYAN)make bootstrap [MIRROR=1]$(RESET)                   Install required system packages (apt)'
	@echo '  $(CYAN)make setup [DEV=1 MIRROR=1]$(RESET)      Install Python dependencies (uv)'
	@echo '  $(CYAN)make install [DEV=1 MIRROR=1]$(RESET)    Install system and Python dependencies'
	@echo '  $(CYAN)make env-secrets$(RESET)                            Create .env and generate API/cookie secrets'
	@echo '  $(CYAN)make data$(RESET)                                  Download and prepare datasets'

	@echo ''
	@echo '$(YELLOW)Code Quality:$(RESET)'
	@echo '  $(CYAN)make check-code$(RESET)     Run ruff linter'
	@echo '  $(CYAN)make format-all$(RESET)     Format code (black + isort + ruff)'
	@echo '  $(CYAN)make typecheck$(RESET)      Run mypy type checker'
	@echo '  $(CYAN)make version-check$(RESET)  Verify README/runtime version markers match pyproject.toml'
	@echo ''
	@echo ''
	@echo '$(CYAN)Docker:$(RESET)'

	@echo '  $(CYAN)make docker-dev$(RESET) [BUILD_ARGS="..."]         Open a dev shell (dev tools included)'
	@echo '  $(CYAN)make docker-train$(RESET) [DEV=1 BUILD_ARGS="..."] Run the configured training jobs'
	@echo '  $(CYAN)make docker-stop$(RESET)                           Stop project containers'
	@echo '  $(CYAN)make docker-manage$(RESET)                         Inspect runtime and project resources'
	@echo '  $(CYAN)make account-disable EMAIL=...$(RESET)             Disable an account and revoke its sessions'
	@echo '  $(CYAN)make account-enable EMAIL=...$(RESET)              Enable an account'
	@echo '  $(CYAN)make account-reset-password EMAIL=...$(RESET)      Reset a password and revoke its sessions'
	@echo '  $(CYAN)make docker-purge-all$(RESET)                      Remove all project containers/images/upload volume'
	@echo '  $(CYAN)make docker-purge-train$(RESET)                    Remove only training/dev containers and images'
	@echo '  $(CYAN)make docker-purge-deploy$(RESET)                   Remove only inference/Nginx containers and inference image'
	@echo '  $(CYAN)make docker-infer$(RESET) [DEPLOYMENT_MODE=proxy] [GPU=1] Start Nginx and FastAPI'
	@echo '  $(CYAN)make local$(RESET)                                 Single-user CPU stack on loopback, no login'
	@echo '  $(CYAN)make docker-deploy-check$(RESET)                   Check Nginx and API services'
	@echo '  $(CYAN)make docker-memory-measure$(RESET) [MEMORY_ARGS="..."] Measure container memory'
	@echo '  Note: inference creates .env if missing; accounts mode needs no shared key.'
	@echo ''
	@echo '$(BLUE)Inference:$(RESET)'
	@echo '  $(CYAN)make docker-batch-test$(RESET) [IMAGES=1] [VIDEOS=1] Container batch inference'
	@echo '  $(CYAN)make benchmark$(RESET) [ROLES=iqa,vqa] [THREADS=2] Held-out accuracy, baseline, latency'
	@echo '  $(CYAN)make pytest$(RESET)          Run the tests/ pytest suite'
	@echo '  $(CYAN)make preflight$(RESET)      Run the full repository gate'

	@echo ''
	@echo '$(BLUE)Maintenance:$(RESET)'
	@echo '  $(CYAN)make cache-clean$(RESET)    Remove cache and temporary files'
	@echo '  $(CYAN)make results-clean$(RESET)       Delete dated .pt/.csv/.log files older than 3 days'
	@echo '  $(CYAN)make archive$(RESET) [ARCHIVE_ARGS="..."]       Package results'
	@echo '  $(CYAN)make git-log$(RESET)        Export git history and update ignore files'
	@echo ''
	@echo '$(CYAN)Info:$(RESET)'
	@echo '  $(CYAN)make info$(RESET)           Show environment details'
	@echo ''

	@echo '$(BOLD)Parameters:$(RESET)'
	@echo '  DEV=1                       Include development tools'
	@echo '  MIRROR=1                    Use the TUNA package mirror'
	@echo '  DEPLOYMENT_MODE=internal|proxy|local  Select start mode (default: internal)'
	@echo '  ROLES=iqa,vqa               Models to benchmark (default: iqa,vqa)'
	@echo '  THREADS=2                   Torch/OpenCV threads for benchmark latency (default: 2)'
	@echo '  BENCH_DEVICE=cpu|cuda       Benchmark device (default: cpu)'
	@echo '  BENCHMARK_ARGS="..."        Advanced benchmark script arguments'
	@echo '  GPU=1                      Enable NVIDIA GPU access (default: CPU)'
	@echo '  BUILD_ARGS="..."             Advanced Docker build flags'
	@echo '  ARCHIVE_ARGS="..."           Advanced archive script arguments'
	@echo '  GIT_LOG_FILE="..."            Git history output path'
	@echo '  BOOTSTRAP_ARGS/SETUP_ARGS    Extra script arguments (optional)'
	@echo ''
	@echo '$(BOLD)Examples:$(RESET)'
	@echo '  $(CYAN)make bootstrap MIRROR=1$(RESET)'
	@echo '  $(CYAN)make setup DEV=1 MIRROR=1$(RESET)'
	@echo '  $(CYAN)make install DEV=1 MIRROR=1$(RESET)'
	@echo '  $(CYAN)make env-secrets$(RESET)'
	@echo '  $(CYAN)make docker-infer$(RESET)'
	@echo '  $(CYAN)make docker-infer DEPLOYMENT_MODE=proxy$(RESET)'
	@echo '  $(CYAN)make local$(RESET)'
	@echo '  $(CYAN)make docker-train DEV=1$(RESET)'
	@echo '  $(CYAN)make docker-train BUILD_ARGS="--build-arg USE_BUILD_PROXY=true"$(RESET)'
	@echo '  $(CYAN)make archive ARCHIVE_ARGS="--results"$(RESET)'
	@echo '  $(CYAN)make git-log GIT_LOG_FILE="results/git_log.txt"$(RESET)'
	@echo '  $(CYAN)make docker-batch-test$(RESET)'
	@echo '  $(CYAN)make docker-batch-test IMAGES=1$(RESET)'
	@echo '  $(CYAN)make docker-batch-test VIDEOS=1$(RESET)'
	@echo '  uv run python -m src.main --dataset tid2013 --model swin_iqa'
	@echo '  uv run python -m src.main --dataset konvid-1k --model swin_vqa'
	@echo ''
	@echo '  Detailed help: make help-TARGET (for example: make help-docker-infer)'
	@echo '  Documentation:  docs/README.md'

h ?:
	@$(MAKE) help

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
			echo 'COMMAND: make docker-infer [DEPLOYMENT_MODE=internal|proxy|local] [GPU=1]'; \
			echo 'PURPOSE: Start vqa-infer and vqa-nginx; default mode is internal.'; \
			echo 'NOTE: internal accounts register via browser; legacy api_key mode requires secrets.' ;; \
		docker-stop) \
			echo 'COMMAND: make docker-stop'; \
			echo 'PURPOSE: Stop project services without deleting data.' ;; \
		docker-purge-all) \
			echo 'COMMAND: make docker-purge-all'; \
			echo 'PURPOSE: Remove project containers, images, network, and named upload volume.'; \
			echo 'WARNING: Does not remove host bind-mounted data; upload volume data is lost.' ;; \
		docker-purge-train) \
			echo 'COMMAND: make docker-purge-train'; \
			echo 'PURPOSE: Remove only vqa-train/vqa-dev containers and their project images.'; \
			echo 'NOTE: Host datasets, results, reports, checkpoints, and volumes are kept.' ;; \
		docker-purge-deploy) \
			echo 'COMMAND: make docker-purge-deploy'; \
			echo 'PURPOSE: Remove only vqa-nginx/vqa-infer containers and the vqa-infer image.'; \
			echo 'NOTE: Host reports, accounts, checkpoints, and upload volume are kept.' ;; \
		docker-deploy-check) \
			echo 'COMMAND: make docker-deploy-check'; \
			echo 'PURPOSE: Check API and Nginx health and the proxied browser entry point.' ;; \
		docker-memory-measure) \
			echo 'COMMAND: make docker-memory-measure [MEMORY_ARGS="..."]'; \
			echo 'PURPOSE: Measure idle, peak, and settled container memory across real requests.'; \
			echo 'OUTPUT: reports/memory/latest.json and reports/memory/latest.md.' ;; \
		docker-batch-test) \
			echo 'COMMAND: make docker-batch-test [IMAGES=1] [VIDEOS=1]'; \
			echo 'PURPOSE: Run batch media inference through deploy/inference/cli.py inside vqa-infer.'; \
			echo 'NOTE: No selector runs both image and video batches.' ;; \
		pytest) echo 'make pytest'; echo '  Run the tests/ pytest suite.' ;; \
		*) echo "No detailed help is available for '$*'. Run 'make help' to list targets."; exit 1 ;; \
		esac

version-sync:
	@$(PYTHON) scripts/sync_version.py

version-check:
	@$(PYTHON) scripts/sync_version.py --check



DEV ?= 0
MIRROR ?= 0
INSTALL_ARGS ?=
BOOTSTRAP_ARGS ?= $(if $(filter 1 true yes,$(MIRROR)),--mirror) $(filter --mirror,$(INSTALL_ARGS))
SETUP_ARGS ?= $(if $(filter 1 true yes,$(MIRROR)),--mirror) $(if $(filter 1 true yes,$(DEV)),--dev) $(INSTALL_ARGS)
ARCHIVE_ARGS ?= --all
GIT_LOG_FILE ?= $(ROOT_DIR)/git_log.txt
bootstrap:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@set -o pipefail; cd $(ROOT_DIR)/scripts && bash bootstrap.sh $(BOOTSTRAP_ARGS) 2>&1 | tee $(LOG_DIR)/bootstrap.log
	@echo "$(GREEN)[OK]$(RESET) Bootstrap complete."



setup:
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@set -o pipefail; cd $(ROOT_DIR)/scripts && bash setup_env.sh $(SETUP_ARGS) 2>&1 | tee $(LOG_DIR)/setup_env.log
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
	@set -o pipefail; cd $(ROOT_DIR)/scripts && bash manage_data.sh 2>&1 | tee $(LOG_DIR)/manage_data.log
	@echo "$(GREEN)[OK]$(RESET) Data ready."




cache-clean:
	@echo "$(YELLOW)[WARN]$(RESET) Cleaning caches..."
	@read -p "Remove all caches? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		set -o pipefail; \
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
	while IFS= read -r -d '' output_file; do \
		filename="$$(basename "$$output_file")"; \
		run_at="$${filename:0:15}"; \
		if [[ "$$run_at" =~ ^[0-9]{8}_[0-9]{6}$$ ]] && [[ "$$run_at" < "$$retention_before" ]]; then \
			printf "$(YELLOW)[INFO]$(RESET) Deleting output file from %s: %s\n" "$$run_at" "$$filename"; \
			rm -f -- "$$output_file"; \
			printf '%s\n' "$$output_file" | tee -a "$(LOG_DIR)/results_clean.log"; \
			count=$$((count + 1)); \
		fi; \
	done < <(find "$(ROOT_DIR)/results" -type f \( -name '*.pt' -o -name '*.csv' -o -name '*.log' \) -print0); \
	echo "$(GREEN)[OK]$(RESET) Deleted $$count dated result file(s) older than 3 days. Log: $(CYAN)$(LOG_DIR)/results_clean.log$(RESET)"


archive:
	@echo "$(BLUE)[INFO]$(RESET) Archiving results..."
	@if [ -f "$(LOG_DIR)/archive.log" ]; then \
		mv "$(LOG_DIR)/archive.log" "$(LOG_DIR)/archive.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@set -o pipefail; cd $(ROOT_DIR)/scripts && bash archive_results.sh $(ARCHIVE_ARGS) 2>&1 | tee $(LOG_DIR)/archive.log
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

preflight:
	@echo "$(BLUE)[INFO]$(RESET) Running the repository gate..."
	@bash scripts/preflight.sh

_docker-batch-images:
	@echo "[INFO] Running image batch inference..."
	@$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) run --rm --no-deps -T vqa-infer \
		.venv/bin/python -m deploy.inference.cli -i examples/images/
	@echo "[OK] Results saved under reports/iqa-test/"

_docker-batch-videos:
	@echo "[INFO] Running video batch inference..."
	@$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) run --rm --no-deps -T vqa-infer \
		.venv/bin/python -m deploy.inference.cli -i examples/videos/
	@echo "[OK] Results saved under reports/vqa-test/"

docker-batch-test:
	$(call check_runtime)
	$(call check_deployment_mode)
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-infer
	@if [ "$(IMAGES)$(VIDEOS)" != "" ] && [ "$(IMAGES)" != "1" ] && [ "$(IMAGES)" != "true" ] && [ "$(IMAGES)" != "yes" ] && [ "$(VIDEOS)" != "1" ] && [ "$(VIDEOS)" != "true" ] && [ "$(VIDEOS)" != "yes" ]; then \
		echo "ERROR: IMAGES and VIDEOS must be boolean values (1/true/yes)."; exit 2; \
	fi
	@if [ "$(IMAGES)$(VIDEOS)" = "" ] || [ "$(IMAGES)" = "1" ] || [ "$(IMAGES)" = "true" ] || [ "$(IMAGES)" = "yes" ]; then \
		$(MAKE) _docker-batch-images; \
	fi
	@if [ "$(IMAGES)$(VIDEOS)" = "" ] || [ "$(VIDEOS)" = "1" ] || [ "$(VIDEOS)" = "true" ] || [ "$(VIDEOS)" = "yes" ]; then \
		$(MAKE) _docker-batch-videos; \
	fi
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
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync ruff format --check . 2>&1 | sed 's/^/  /'
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync ruff check . 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) All checks passed."

fmt:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting code with ruff..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,ruff,dev)
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync ruff format . 2>&1 | sed 's/^/  /'
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync ruff check . --fix 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Formatting complete."

black:
	@echo "$(YELLOW)[INFO]$(RESET) Formatting with black..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,black,dev)
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync black src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) black complete."

isort:
	@echo "$(YELLOW)[INFO]$(RESET) Sorting imports with isort..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,isort,dev)
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync isort src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) isort complete."

format-all: black isort fmt
	@echo "$(GREEN)[OK]$(RESET) All formatters completed."

typecheck:
	@echo "$(YELLOW)[INFO]$(RESET) Running mypy type checks..."
	@[ -d "$(ROOT_DIR)/.venv" ] || (echo "$(RED)[ERROR]$(RESET) Virtual env not found. Run 'make setup'." && exit 1)
	$(call require_tool,mypy,dev)
	@set -o pipefail; cd $(ROOT_DIR) && uv run --no-sync mypy --explicit-package-bases deploy src --ignore-missing-imports 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Type checks complete."

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
	else \
		echo "$(YELLOW)[WARN] Virtual environment not found. Run 'make setup' first.$(RESET)"; \
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
	@echo "  uv run python -m src.main --dataset tid2013 --model swin_iqa"
	@echo "  uv run python -m src.main --dataset konvid-1k --model swin_vqa"
	@echo "========================================================================"




# ============================================================
# Docker Commands
# ============================================================
COMPOSE := $(shell \
	if command -v podman >/dev/null 2>&1 \
		&& command -v podman-compose >/dev/null 2>&1; then \
		echo "podman-compose"; \
	elif command -v podman >/dev/null 2>&1 \
		&& podman compose version >/dev/null 2>&1; then \
		echo "podman compose"; \
	elif command -v docker >/dev/null 2>&1 \
		&& command -v docker-compose >/dev/null 2>&1; then \
		echo "docker-compose"; \
	elif command -v docker >/dev/null 2>&1 \
		&& docker compose version >/dev/null 2>&1; then \
		echo "docker compose"; \
	else \
		echo ""; \
	fi)

RUNTIME := $(if $(filter podman%,$(COMPOSE)),podman,$(if $(filter docker%,$(COMPOSE)),docker,))

COMPOSE_FILES := -f deploy-config/compose/docker-compose.yaml
GPU ?= 0
GPU_DISABLED := $(filter 0 false no,$(strip $(GPU)))

ifeq ($(RUNTIME),podman)
ifneq ($(GPU_DISABLED),)
COMPOSE_FILES += -f deploy-config/compose/docker-compose.podman-cpu.yaml
else
COMPOSE_FILES += -f deploy-config/compose/docker-compose.podman-gpu.yaml
endif
endif

ifeq ($(RUNTIME),docker)
ifneq ($(GPU_DISABLED),)
COMPOSE_FILES += -f deploy-config/compose/docker-compose.docker-cpu.yaml
else
COMPOSE_FILES += -f deploy-config/compose/docker-compose.docker-gpu.yaml
endif
endif

COMPOSE_ENV = DEEP_VQA_DEPLOYMENT_CONFIG="$(DEPLOYMENT_CONFIG)" \
	IQA_MODEL_VOLUME_PATH="$(ROOT_DIR)/deploy/iqa-models" \
	VQA_MODEL_VOLUME_PATH="$(ROOT_DIR)/deploy/vqa-models"


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
		echo "[INFO] Internal accounts mode needs no shared API key; legacy api_key mode requires secrets."; \
	fi
endef

define show_deployment_context
	@echo "$(CYAN)[INFO]$(RESET) Deployment mode: $(1)"
	@echo "$(CYAN)[INFO]$(RESET) Configuration file: $(2)"
	@echo "$(CYAN)[INFO]$(RESET) Login mode: $(3)"
	@echo "$(CYAN)[INFO]$(RESET) History storage: $(4)"
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

define check_deployment_mode
	@if [ "$(DEPLOYMENT_MODE)" != internal ] && [ "$(DEPLOYMENT_MODE)" != proxy ] && [ "$(DEPLOYMENT_MODE)" != local ]; then \
		echo "ERROR: DEPLOYMENT_MODE must be internal, proxy, or local"; \
		exit 2; \
	fi
endef

# Wait only while a container can still become ready. Exited/dead/unhealthy
# states fail immediately instead of consuming the entire timeout budget.
define wait_container_http
	@attempts=$(3); unconfigured_attempts=0; \
	for i in $$(seq 1 $$attempts); do \
		state=$$($(RUNTIME) inspect --format '{{.State.Status}}' $(1) 2>/dev/null || echo missing); \
		health=$$($(RUNTIME) inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unconfigured{{end}}' $(1) 2>/dev/null || echo missing); \
		case "$$state/$$health" in \
			missing/*) echo "$(RED)[ERROR]$(RESET) $(1) is not present."; exit 1 ;; \
			exited/*|dead/*|removing/*|*/unhealthy) echo "$(RED)[ERROR]$(RESET) $(1) cannot become ready: state=$$state health=$$health"; exit 1 ;; \
			running/healthy) ;; \
			running/unconfigured) unconfigured_attempts=$$((unconfigured_attempts + 1)) ;; \
			created/*|running/starting) ;; \
			*) unconfigured_attempts=$$((unconfigured_attempts + 1)) ;; \
		esac; \
		probe_ok=1; \
		curl --noproxy '*' -fsS --max-time 5 "$(2)" >/dev/null 2>&1 && probe_ok=0; \
		if [ "$$probe_ok" -eq 0 ]; then echo "$(GREEN)[OK]$(RESET) $(1) is healthy."; exit 0; fi; \
		if [ "$$unconfigured_attempts" -ge 2 ]; then echo "$(RED)[ERROR]$(RESET) $(1) endpoint failed after two probes."; exit 1; fi; \
		[ "$$i" -lt "$$attempts" ] && sleep 2; \
	done; \
	echo "$(RED)[ERROR]$(RESET) $(1) did not become healthy."; exit 1
endef


BUILD_ARGS ?=
DEPLOYMENT_MODE ?= internal
DEPLOYMENT_CONFIG ?= deploy-config/profiles/infer_deploy.$(DEPLOYMENT_MODE).yaml
API_HEALTH_URL = http://127.0.0.1:$${API_PORT:-8001}/v1/health
WEB_HEALTH_URL = http://127.0.0.1:$${WEB_PORT:-8000}/api/v1/health


docker-dev:
	$(call check_runtime)
	$(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) --build-arg INSTALL_DEV=true vqa-dev
	$(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-dev vqa-dev


docker-train:
	$(call check_runtime)
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) --build-arg INSTALL_DEV=$(if $(filter 1 true yes,$(DEV)),true,false) vqa-train
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) run --rm --name vqa-train vqa-train bash -c "\
		set -o pipefail && \
		make data && \
		uv run python -m src.main --dataset tid2013 --model swin_iqa 2>&1 | tee /app/results/tid2013.log && \
		uv run python -m src.main --dataset konvid-1k --model swin_vqa 2>&1 | tee /app/results/konvid-1k.log \
	"


_docker-api:
	$(call check_runtime)
	$(call ensure_env_file)
	$(call check_deploy_auth)
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) build $(BUILD_ARGS) vqa-infer
	@# Remove the dependent proxy before replacing vqa-infer. Rootless Podman
	@# refuses to remove a container while another container depends on it.
	@$(RUNTIME) rm -f vqa-nginx 2>/dev/null || true
	@$(RUNTIME) rm -f vqa-infer 2>/dev/null || true
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) up -d vqa-infer
	@echo "[INFO] Waiting for FastAPI health at $(API_HEALTH_URL)..."
	$(call wait_container_http,vqa-infer,$(API_HEALTH_URL),60)


_docker-infer: _docker-api
	@$(RUNTIME) rm -f vqa-nginx 2>/dev/null || true
	$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) up -d vqa-nginx
	@echo "[INFO] Waiting for Nginx and proxied API..."
	$(call wait_container_http,vqa-nginx,$(WEB_HEALTH_URL),30)
	@echo "$(GREEN)[OK]$(RESET) Inference stack is ready: http://127.0.0.1:$${WEB_PORT:-8000}/"

# User command for starting Nginx and FastAPI.
docker-infer:
	$(call check_deployment_mode)
	$(call run_infer_mode,$(DEPLOYMENT_CONFIG),$(if $(filter internal,$(DEPLOYMENT_MODE)),accounts,none),$(DEPLOYMENT_MODE),$(if $(filter proxy,$(DEPLOYMENT_MODE)),none,sqlite))

# Single-user stack for a personal machine: CPU only, no login, loopback bind.
local:
	@$(MAKE) --no-print-directory docker-infer DEPLOYMENT_MODE=local GPU=0

# Account operations run inside the inference container: they change the live
# accounts database and need the service's dependencies, neither of which exists
# on the deployment host.
account-disable:
	@test -n "$(EMAIL)" || (echo "ERROR: set EMAIL=user@example.com"; exit 1)
	$(call check_runtime)
	@$(RUNTIME) exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account disable "$(EMAIL)"

account-enable:
	@test -n "$(EMAIL)" || (echo "ERROR: set EMAIL=user@example.com"; exit 1)
	$(call check_runtime)
	@$(RUNTIME) exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account enable "$(EMAIL)"

account-reset-password:
	@test -n "$(EMAIL)" || (echo "ERROR: set EMAIL=user@example.com"; exit 1)
	$(call check_runtime)
	@$(RUNTIME) exec -it vqa-infer .venv/bin/python -m deploy.web.manage_account reset-password "$(EMAIL)"

# Validate the selected mode configuration before starting the common services.
define run_infer_mode
	$(call show_deployment_context,$(3),$(1),$(2),$(4))
	@if [ "$(3)" = "internal" ]; then \
		echo "$(YELLOW)[INFO]$(RESET) Internal mode signs in with email + password."; \
		echo "$(YELLOW)[INFO]$(RESET) Open the browser entry point and register the first account."; \
	fi
	@if [ "$(3)" = "local" ]; then \
		echo "$(YELLOW)[INFO]$(RESET) Local mode has no login and keeps history on this machine."; \
		echo "$(YELLOW)[INFO]$(RESET) The stack listens on 127.0.0.1 only."; \
	fi
	@if ! grep -Eq '^[[:space:]]*mode:[[:space:]]*$(2)([[:space:]]|$$)' $(1); then \
		echo "ERROR: $(3) mode requires auth.mode=$(2)"; exit 1; \
	fi
	@if ! grep -Eq '^[[:space:]]*backend:[[:space:]]*$(4)([[:space:]]|$$)' $(1); then \
		echo "ERROR: $(3) mode requires evaluation_store.backend=$(4)"; exit 1; \
	fi
	$(MAKE) _docker-infer DEPLOYMENT_CONFIG=$(1)
endef

docker-stop:
	$(call check_runtime)
	@echo "[INFO] Stopping containers..."
	@$(RUNTIME) stop vqa-dev 2>/dev/null || true
	@$(RUNTIME) stop vqa-infer 2>/dev/null || true
	@$(RUNTIME) stop vqa-nginx 2>/dev/null || true
	@$(RUNTIME) stop vqa-train 2>/dev/null || true
	@echo "$(GREEN)[OK]$(RESET) Containers stopped."

docker-deploy-check:
	@if [ -z "$(RUNTIME)" ] || ! $(RUNTIME) container inspect vqa-infer >/dev/null 2>&1; then \
		echo "$(RED)[ERROR]$(RESET) No containerized inference stack found. Run 'make docker-infer'."; \
		exit 1; \
	fi; \
	failed=0; logs_needed=0; \
	check_http() { \
		name="$$1"; url="$$2"; optional="$$3"; max_attempts="$$4"; \
		if ! $(RUNTIME) container inspect "$$name" >/dev/null 2>&1; then \
			if [ "$$optional" = yes ]; then echo "[INFO] $$name (optional): not present"; return 0; fi; \
			echo "$(RED)[ERROR]$(RESET) $$name is not present."; return 1; \
		fi; \
		state=$$($(RUNTIME) inspect --format '{{.State.Status}}' "$$name"); \
		health=$$($(RUNTIME) inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unconfigured{{end}}' "$$name"); \
		[ -n "$$health" ] || health=unconfigured; \
		echo "[INFO] $$name: state=$$state health=$$health"; \
		case "$$state/$$health" in \
			exited/*|dead/*|removing/*|*/unhealthy) echo "$(RED)[ERROR]$(RESET) $$name cannot become ready."; logs_needed=1; return 1 ;; \
			running/healthy) attempts=1 ;; \
			running/unconfigured) attempts=2 ;; \
			created/*|running/starting) attempts="$$max_attempts" ;; \
			*) attempts=2 ;; \
		esac; \
		for i in $$(seq 1 $$attempts); do \
			probe_ok=1; \
			curl --noproxy '*' -fsS --max-time 5 "$$url" >/dev/null 2>&1 && probe_ok=0; \
			if [ "$$probe_ok" -eq 0 ]; then echo "$(GREEN)[OK]$(RESET) $$name endpoint is healthy."; return 0; fi; \
			[ "$$i" -lt "$$attempts" ] && sleep 2; \
		done; \
		echo "$(RED)[ERROR]$(RESET) $$name endpoint failed."; logs_needed=1; return 1; \
	}; \
	check_http vqa-infer "$(API_HEALTH_URL)" no 60 || failed=1; \
	check_http vqa-nginx "$(WEB_HEALTH_URL)" no 30 || failed=1; \
	if [ "$$failed" -ne 0 ]; then \
		if [ "$$logs_needed" -ne 0 ]; then \
			echo "$(YELLOW)[INFO]$(RESET) Recent deployment logs:"; \
			for name in vqa-infer vqa-nginx; do \
				echo "--- $$name ---"; \
				$(RUNTIME) logs --tail=60 "$$name" 2>/dev/null || true; \
			done; \
		fi; \
		exit 1; \
	fi; \
	echo "$(GREEN)[OK]$(RESET) Deployment checks passed."



docker-memory-measure:
	$(call check_runtime)
	@$(PYTHON) scripts/measure_container_memory.py --runtime $(RUNTIME) $(MEMORY_ARGS)

benchmark:
	@echo "$(BLUE)[INFO]$(RESET) Benchmarking serving checkpoints on held-out test splits..."
	@set -o pipefail; cd $(ROOT_DIR) && PYTHONPATH="$(ROOT_DIR)" $(PYTHON) scripts/benchmark_models.py \
		--roles "$(if $(ROLES),$(ROLES),iqa,vqa)" \
		--device "$(if $(BENCH_DEVICE),$(BENCH_DEVICE),cpu)" \
		--threads "$(if $(THREADS),$(THREADS),2)" $(BENCHMARK_ARGS) 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)[OK]$(RESET) Benchmark report written under reports/benchmark/."

docker-manage:
	$(call check_runtime)
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
	@$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) ps 2>/dev/null || echo "  (none)"



docker-purge-train:
	$(call check_runtime)
	@echo "[INFO] Removing only training/development containers and images..."
	@$(RUNTIME) rm -f vqa-train vqa-dev 2>/dev/null || true
	@$(RUNTIME) rmi vqa-train:latest vqa-dev:latest 2>/dev/null || true
	@echo "$(GREEN)[OK]$(RESET) Training/development containers and images removed; bind-mounted data kept."

docker-purge-deploy:
	$(call check_runtime)
	@echo "[INFO] Removing only inference/Nginx containers and the inference image..."
	@$(RUNTIME) rm -f vqa-nginx vqa-infer 2>/dev/null || true
	@$(RUNTIME) rmi vqa-infer:latest 2>/dev/null || true
	@echo "$(GREEN)[OK]$(RESET) Deployment containers and inference image removed; reports and upload volume kept."

docker-purge-all:
	$(call check_runtime)
	@echo "[INFO] Removing project containers, images, network and named upload volume..."
	@$(COMPOSE_ENV) $(COMPOSE) $(COMPOSE_FILES) down --volumes
	@$(RUNTIME) rm -f vqa-train vqa-dev vqa-infer vqa-nginx 2>/dev/null || true
	@$(RUNTIME) rmi vqa-train:latest vqa-dev:latest vqa-infer:latest 2>/dev/null || true

	@echo "$(GREEN)[OK]$(RESET) Docker/Podman purge complete."
