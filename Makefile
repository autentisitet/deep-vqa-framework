# --- Color definitions for output styling ---
GREEN := \033[0;32m
BLUE := \033[0;34m
RED := \033[0;31m
YELLOW := \033[0;33m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
WHITE := \033[1;37m
BOLD := \033[1m
DIM := \033[2m
RESET := \033[0m


# --- Project Metadata ---
PROJECT_NAME := Deep-VQA-Framework
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
$(shell [ -f "$(ROOT_DIR)/Makefile" ] || (echo "$(RED)Error: Makefile not found in ROOT_DIR$(RESET)"; exit 1))

# Automatically determine the operating system environment
# The OS environment variable under Windows is usually Windows_NT
ifeq ($(OS),Windows_NT)
	PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/Scripts/python.exe" ]; then echo "$(ROOT_DIR)/.venv/Scripts/python.exe"; else echo "python"; fi)
	UV_RUN := uv run
else
	PYTHON_CMD := $(shell if [ -f "$(ROOT_DIR)/.venv/bin/python" ]; then echo "$(ROOT_DIR)/.venv/bin/python"; else echo "python3"; fi)
	UV_RUN := uv run
endif

PYTHON := $(PYTHON_CMD)

LOG_DIR := $(ROOT_DIR)/results/scripts_logs
SECURITY_DIR := $(ROOT_DIR)/reports/security

$(shell mkdir -p $(LOG_DIR))
$(shell mkdir -p $(SECURITY_DIR))
$(info 📂 Project Root detected as: $(CYAN)$(ROOT_DIR)$(RESET))


.PHONY: help setup data link check-system \
				optimize clean archive \
				check-code fmt black isort format-all typecheck \
				vuln-audit sbom safety security-all \
				info

.DEFAULT_GOAL := help


help:
	@echo "$(BOLD)$(CYAN)🛠️  $(PROJECT_NAME) Commands:$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)📦 Environment:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)         - Install dependencies and setup environment"
	@echo "  $(GREEN)make data$(RESET)          - Prepare datasets"
	@echo "  $(GREEN)make link$(RESET)          - Set up symbolic links"
	@echo "  $(GREEN)make optimize$(RESET)      - Optimize network/Jupyter settings"
	@echo ""
	@echo "$(BOLD)$(BLUE)🔍 Status:$(RESET)"
	@echo "  $(BLUE)make check-system$(RESET)   - Check system status (GPU/memory/process)"
	@echo "  $(BLUE)make info$(RESET)           - Show environment information"
	@echo ""
	@echo "$(BOLD)$(YELLOW)📝 Code Quality:$(RESET)"
	@echo "  $(YELLOW)make check-code$(RESET)   - Check code style with ruff"
	@echo "  $(YELLOW)make fmt$(RESET)          - Format code with ruff"
	@echo "  $(YELLOW)make black$(RESET)        - Format code with black"
	@echo "  $(YELLOW)make isort$(RESET)        - Sort imports with isort"
	@echo "  $(YELLOW)make format-all$(RESET)   - Run all formatters (black + isort + ruff)"
	@echo "  $(YELLOW)make typecheck$(RESET)    - Perform type checking with mypy"
	@echo ""
	@echo "$(BOLD)$(MAGENTA)🔒 Security:$(RESET)"
	@echo "  $(MAGENTA)make vuln-audit$(RESET)  - Audit dependencies for vulnerabilities"
	@echo "  $(MAGENTA)make sbom$(RESET)        - Generate SBOM (CycloneDX)"
	@echo "  $(MAGENTA)make safety$(RESET)      - Check dependencies with safety"
	@echo "  $(MAGENTA)make security-all$(RESET) - Run all security checks (audit + sbom + safety)"
	@echo ""
	@echo "$(BOLD)$(RED)🧹 Maintenance:$(RESET)"
	@echo "  $(RED)make clean$(RESET)           - Clean caches and temp files"
	@echo "  $(RED)make archive$(RESET)         - Package results"
	@echo ""
	@echo "$(BOLD)$(WHITE)🚀 Run training directly with uv:$(RESET)"
	@echo "  $(DIM)uv run python -m src.main --dataset tid2013 --model resnet_iqa$(RESET)"
	@echo "  $(DIM)DEBUG=1 uv run python -m src.main --dataset tid2013$(RESET)"
	@echo "  $(DIM)nohup uv run python -m src.main ... > train.log 2>&1 &$(RESET)"
	@echo "  $(DIM)tail -f train.log$(RESET)"
	@echo ""


# 1. Environment Initialization
-include $(ROOT_DIR)/Makefile.local
-include $(ROOT_DIR)/config.mk
SETUP_ARGS ?= --mirror
setup:
	@echo "⚙️ $(BLUE)Setting script permissions...$(RESET) (Log: $(LOG_DIR)/setup_env.log)"
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@echo "⚙️ $(BLUE)Setting up environment...$(RESET)"
	@cd $(ROOT_DIR)/scripts && bash setup_env.sh $(SETUP_ARGS) 2>&1 | tee $(LOG_DIR)/setup_env.log
	@echo "$(GREEN)✅ Environment ready$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Setup complete!$(RESET)Run 'make info' to verify."
	@echo ""
	@echo "$(YELLOW)⚠️  IMPORTANT: If 'uv' command is not found in your shell:$(RESET)"
	@echo "   export PATH=\"\$$HOME/.local/bin:\$$PATH\""
	@echo "   Add to ~/.bashrc: echo 'export PATH=\"\$$HOME/.local/bin:\$$PATH\"' >> ~/.bashrc"
	@echo ""


# 2. Data Preparation
data:
	@echo "📦 $(BLUE)Preparing datasets...$(RESET) (Log: $(LOG_DIR)/manage_data.log)"
	@cd $(ROOT_DIR)/scripts && bash manage_data.sh 2>&1 | tee $(LOG_DIR)/manage_data.log
	@echo "$(GREEN)✅ Data ready$(RESET)"

# 3. Symbolic Links
link:
	@echo "🔗 $(BLUE)Setting up symbolic links...$(RESET) (Log: $(LOG_DIR)/setup_links.log)"
	@cd $(ROOT_DIR)/scripts && bash setup_links.sh --all 2>&1 | tee $(LOG_DIR)/setup_links.log	
	@echo "$(GREEN)✅ Symbolic links ready$(RESET)"


# 4. Status Audit
check-system:
	@echo "🔍 $(BLUE)Checking system status...$(RESET) (Log: $(LOG_DIR)/system_check.log)"
	@cd $(ROOT_DIR)/scripts && bash system_check.sh 2>&1 | tee $(LOG_DIR)/system_check.log	
	@echo "$(GREEN)✅ System check completed$(RESET)"


# 5. Optimize Environment (network/Jupyter settings)
optimize:
	@echo "🔧 $(BLUE)Optimizing Environment...$(RESET) (Log: $(LOG_DIR)/optimize_env.log)"
	@cd $(ROOT_DIR)/scripts && bash optimize_env.sh 2>&1 | tee $(LOG_DIR)/optimize_env.log
	@echo "$(GREEN)✅ Optimization completed$(RESET)"


# 6. Clear cache
clean:
	@echo "🧹 $(RED)Cleaning caches...$(RESET)"
	@read -p "$(YELLOW)⚠️ Are you sure you want to clean all caches? [y/N] $(RESET)" confirm; \
	if [ "$$confirm" = "y" ]; then \
		echo "🧹 $(RED)Cleaning...$(RESET) (Log: $(LOG_DIR)/cache_clean.log)"; \
		bash $(ROOT_DIR)/scripts/cache_clean.sh 2>&1 | tee $(LOG_DIR)/cache_clean.log; \
		echo "$(GREEN)✅ Clean completed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️ Clean aborted.$(RESET)"; \
	fi


# 7. Packaging Results
archive:
	@echo "📦 $(BLUE)Archiving...$(RESET) (Log: $(LOG_DIR)/archive.log)"
	@if [ -f "$(LOG_DIR)/archive.log" ]; then \
		mv "$(LOG_DIR)/archive.log" "$(LOG_DIR)/archive.log.$$(date +%Y%m%d%H%M%S).bak"; \
	fi
	@cd $(ROOT_DIR)/scripts && bash archive_results.sh --all 2>&1 | tee $(LOG_DIR)/archive.log
	@echo "$(GREEN)✅ Archive completed$(RESET)"



# ============================================================
# Code Quality
# ============================================================
# 8. Validate code style and specifications
check-code:
	@echo "📝 $(YELLOW)Verifying code style and specifications...$(RESET)"
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show ruff >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ ruff not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"' first.$(RESET)"; \
		exit 1; \
	fi
	@cd $(ROOT_DIR) && uv run ruff format --check . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)✅ Verification completed, code quality is good!$(RESET)"


# 9. Format code with ruff
fmt:
	@echo "📝 $(YELLOW)Formatting code...$(RESET)"
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show ruff >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ ruff not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"' first.$(RESET)"; \
		exit 1; \
	fi
	@cd $(ROOT_DIR) && uv run ruff format . 2>&1 | sed 's/^/  /'
	@cd $(ROOT_DIR) && uv run ruff check . --fix 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)✅ Code formatted successfully!$(RESET)"


# 10. Format code with black
black:
	@echo "🖤 $(YELLOW)Formatting code with black...$(RESET)"
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show black >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ black not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"' first.$(RESET)"; \
		exit 1; \
	fi
	@cd $(ROOT_DIR) && uv run black src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)✅ Code formatted with black!$(RESET)"


# 11. Sort imports with isort
isort:
	@echo "📋 $(YELLOW)Sorting imports with isort...$(RESET)"
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show isort >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ isort not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"' first.$(RESET)"; \
		exit 1; \
	fi
	@cd $(ROOT_DIR) && uv run isort src/ 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)✅ Imports sorted with isort!$(RESET)"


# 12. Run all formatters
format-all: black isort fmt
	@echo "$(GREEN)✅ All formatters completed!$(RESET)"
	@echo ""
	@echo "📊 Summary:"
	@echo "  $(GREEN)•$(RESET) black (code formatting)"
	@echo "  $(GREEN)•$(RESET) isort (import sorting)"
	@echo "  $(GREEN)•$(RESET) ruff (code formatting & fixing)"


# 13. Type checking with mypy
typecheck:
	@echo "🔍 $(YELLOW)Performing type checking with mypy...$(RESET)"
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show mypy >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ mypy not installed. Run 'make setup SETUP_ARGS=\"--mirror --dev\"' first.$(RESET)"; \
		exit 1; \
	fi
	@cd $(ROOT_DIR) && uv run mypy src/ --ignore-missing-imports 2>&1 | sed 's/^/  /'
	@echo "$(GREEN)✅ Type checking completed!$(RESET)"


	


# ============================================================
# Security
# ============================================================
# 14. Audit dependencies for vulnerabilities
vuln-audit:
	@echo "🔒 $(MAGENTA)Auditing dependencies for vulnerabilities...$(RESET)"
	@echo "📂 Reports saved to: $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo ""
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show pip-audit >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ pip-audit not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"' first.$(RESET)"; \
		exit 1; \
	fi

	@echo "📦 Generating dependency list..."
	@cd $(ROOT_DIR) && uv pip freeze > $(ROOT_DIR)/requirements.txt
	@echo "🔍 $(MAGENTA)Running pip-audit...$(RESET)"
	@echo ""
	@cd $(ROOT_DIR) && uv run pip-audit \
		--requirement $(ROOT_DIR)/requirements.txt \
		--format json \
		--output $(SECURITY_DIR)/audit-report.json \
		--desc || true
	@cd $(ROOT_DIR) && uv run pip-audit \
		--requirement $(ROOT_DIR)/requirements.txt \
		--format columns \
		| tee $(SECURITY_DIR)/audit-report.txt
	@echo ""
	@echo "$(GREEN)✅ Audit complete.$(RESET)"
	@echo "📊 Reports generated in $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo "  $(BOLD)JSON:$(RESET) $(CYAN)$(SECURITY_DIR)/audit-report.json$(RESET)"
	@echo "  $(BOLD)Text:$(RESET)  $(CYAN)$(SECURITY_DIR)/audit-report.txt$(RESET)"
	@echo ""
	@if grep -v "deep-vqa-framework" $(SECURITY_DIR)/audit-report.txt | grep -q "No known vulnerabilities found"; then \
    	echo "$(GREEN)✅ No vulnerabilities found!$(RESET)"; \
	else \
		echo "$(RED)☢️ Vulnerabilities detected! Check the reports above.$(RESET)"; \
	fi


# 15. Generate SBOM (Software Bill of Materials)
sbom:
	@echo "📋 $(MAGENTA)Generating Software Bill of Materials (SBOM)...$(RESET)"
	@echo "📂 Reports saved to: $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo ""
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run cyclonedx-py --version >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ cyclonedx-bom not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"' first.$(RESET)"; \
		exit 1; \
	fi


	@echo "📦 Generating CycloneDX JSON SBOM..."
	@cd $(ROOT_DIR) && uv run cyclonedx-py environment \
		--output-format json \
		--output-file $(SECURITY_DIR)/sbom-cyclonedx.json 2>&1 | sed 's/^/  /'
	@echo "📦 Generating CycloneDX XML SBOM..."
	@cd $(ROOT_DIR) && uv run cyclonedx-py environment \
		--output-format xml \
		--output-file $(SECURITY_DIR)/sbom-cyclonedx.xml 2>&1 | sed 's/^/  /'
	@echo "📦 Generating dependency list..."
	@cd $(ROOT_DIR) && uv pip freeze > $(SECURITY_DIR)/dependencies.txt


	@echo "📦 Generating dependency list..."
	@cd $(ROOT_DIR) && uv pip freeze > $(SECURITY_DIR)/dependencies.txt

	@echo "🌳 Generating dependency tree..."
	@cd $(ROOT_DIR) && uv pip tree > $(SECURITY_DIR)/dependency-tree.txt 2>/dev/null || echo "  $(YELLOW)⚠️ pip tree not available$(RESET)"

	@echo ""
	@echo "$(GREEN)✅ SBOM generation complete!$(RESET)"
	@echo "📊 Reports generated in $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo "  $(BOLD)CycloneDX JSON:$(RESET) $(CYAN)$(SECURITY_DIR)/sbom-cyclonedx.json$(RESET)"
	@echo "  $(BOLD)CycloneDX XML:$(RESET)  $(CYAN)$(SECURITY_DIR)/sbom-cyclonedx.xml$(RESET)"
	@echo "  $(BOLD)Dependencies:$(RESET)   $(CYAN)$(SECURITY_DIR)/dependencies.txt$(RESET)"
	@echo "  $(BOLD)Dependency Tree:$(RESET) $(CYAN)$(SECURITY_DIR)/dependency-tree.txt$(RESET)"
	@echo ""
	@echo "ℹ️ $(DIM)SBOM files can be used with security tools like:$(RESET)"
	@echo "  $(DIM)- OWASP Dependency Check$(RESET)"
	@echo "  $(DIM)- Snyk$(RESET)"
	@echo "  $(DIM)- GitHub Dependabot$(RESET)"



# 16. Safety check for known vulnerabilities
safety:
	@echo "🛡️ $(MAGENTA)Running safety scan...$(RESET)"
	@echo "📂 Reports saved to: $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo ""
	@if [ ! -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! uv run pip show safety >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️ safety not installed. Run 'make setup SETUP_ARGS=\"--mirror --security\"' first.$(RESET)"; \
		exit 1; \
	fi

	@echo "📦 Generating dependency list..."
	@cd $(ROOT_DIR) && uv pip freeze > $(ROOT_DIR)/requirements.txt
	@echo "🔍 $(MAGENTA)Running safety...$(RESET)"
	@echo ""
	@cd $(ROOT_DIR) && uv run safety scan \
		--full-report \
		2>&1 | tee $(SECURITY_DIR)/safety-report.txt || true
	@echo ""
	@echo "$(GREEN)✅ Safety scan complete.$(RESET)"
	@echo "📊 Reports generated in $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo "  $(BOLD)Text:$(RESET)  $(CYAN)$(SECURITY_DIR)/safety-report.txt$(RESET)"
	@echo ""
	@if grep -q "No known vulnerabilities" $(SECURITY_DIR)/safety-report.txt 2>/dev/null; then \
		echo "$(GREEN)✅ No vulnerabilities found!$(RESET)"; \
	else \
		echo "$(RED)☢️ Vulnerabilities detected! Check the reports above.$(RESET)"; \
	fi



# 17. Run all security checks
security-all: vuln-audit sbom safety
	@echo ""
	@echo "🔒 $(GREEN)All security checks completed!$(RESET)"
	@echo "📂 Reports saved to: $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo ""
	@echo "📊 Summary:"
	@echo "  $(BOLD)Vulnerability Report:$(RESET) $(CYAN)$(SECURITY_DIR)/audit-report.json$(RESET)"
	@echo "  $(BOLD)SBOM:$(RESET)               $(CYAN)$(SECURITY_DIR)/sbom-cyclonedx.json$(RESET)"
	@echo "  $(BOLD)Safety Report:$(RESET)      $(CYAN)$(SECURITY_DIR)/safety-report.json$(RESET)"
	@echo ""
	@echo "ℹ️ $(DIM)To view vulnerability details:$(RESET)"
	@echo "  $(DIM)cat $(SECURITY_DIR)/audit-report.txt$(RESET)"
	@echo "  $(DIM)jq . $(SECURITY_DIR)/audit-report.json | less$(RESET)"




# ============================================================
# 18. Show current environment info
# ============================================================
info:
	@echo "$(BOLD)$(CYAN)📊 Environment Information:$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)📁 Project:$(RESET)"
	@echo "  $(BOLD)Project Root:$(RESET)     $(CYAN)$(ROOT_DIR)$(RESET)"
	@echo "  $(BOLD)Project Name:$(RESET)     $(CYAN)$(PROJECT_NAME)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)🐍 Python:$(RESET)"
	@echo "  $(BOLD)Python:$(RESET)           $(CYAN)$(PYTHON)$(RESET)"
	@echo "  $(BOLD)Python Version:$(RESET)   $$($(PYTHON) --version 2>/dev/null || echo '$(RED)Not found$(RESET)')"
	@echo "  $(BOLD)uv Version:$(RESET)       $$(cd $(ROOT_DIR) && $(HOME)/.local/bin/uv --version 2>/dev/null | cut -d' ' -f2 || echo '$(RED)Not found$(RESET)')"
	@echo "  $(BOLD)Pip Version:$(RESET)      $$(cd $(ROOT_DIR) && uv run pip --version 2>/dev/null | cut -d' ' -f2 || echo '$(RED)Not found$(RESET)')"
	@echo ""
	@echo "$(BOLD)$(BLUE)📂 Directories:$(RESET)"
	@echo "  $(BOLD)Log Directory:$(RESET)    $(CYAN)$(LOG_DIR)$(RESET)"
	@echo "  $(BOLD)Security Reports:$(RESET) $(CYAN)$(SECURITY_DIR)$(RESET)"
	@echo "  $(BOLD)Cache Directory:$(RESET)  $(CYAN)$$HOME/.cache/uv$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)⚙️  Configuration:$(RESET)"
	@echo "  $(BOLD)SETUP_ARGS:$(RESET)       $(CYAN)$(SETUP_ARGS)$(RESET)"
	@if [ -f "$(ROOT_DIR)/Makefile.local" ]; then \
		echo "  $(BOLD)Makefile.local:$(RESET)   $(GREEN)✅ Loaded$(RESET)"; \
	elif [ -f "$(ROOT_DIR)/config.mk" ]; then \
		echo "  $(BOLD)config.mk:$(RESET)        $(GREEN)✅ Loaded$(RESET)"; \
	else \
		echo "  $(BOLD)Local config:$(RESET)     $(YELLOW)⚠️ Not found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)🖥️  System:$(RESET)"
	@echo "  $(BOLD)OS:$(RESET)               $(CYAN)$$(uname -s) $$(uname -r)$(RESET)"
	@echo "  $(BOLD)Architecture:$(RESET)     $(CYAN)$$(uname -m)$(RESET)"
	@echo "  $(BOLD)Hostname:$(RESET)         $(CYAN)$$(hostname)$(RESET)"
	@echo "  $(BOLD)User:$(RESET)             $(CYAN)$$(whoami)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(BLUE)💻 Hardware:$(RESET)"
	@echo "  $(BOLD)CPU:$(RESET)              $(CYAN)$$(nproc) cores$(RESET)"
	@echo "  $(BOLD)Memory:$(RESET)           $(CYAN)$$(free -h | awk '/^Mem:/ {print $$2}') total, $$(free -h | awk '/^Mem:/ {print $$4}') available$(RESET)"
	@echo "  $(BOLD)Disk:$(RESET)             $(CYAN)$$(df -h $(ROOT_DIR) | awk 'NR==2 {print $$4}') available$(RESET)"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "  $(BOLD)GPU:$(RESET)              $(CYAN)$$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)$(RESET)"; \
		echo "  $(BOLD)GPU Memory:$(RESET)        $(CYAN)$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) total$(RESET)"; \
		echo "  $(BOLD)Driver Version:$(RESET)    $(CYAN)$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)$(RESET)"; \
		echo "  $(BOLD)CUDA Version:$(RESET)      $(CYAN)$$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | sed 's/.*CUDA Version: *//' || echo 'N/A')$(RESET)"; \
	else \
		echo "  $(BOLD)GPU:$(RESET)              $(YELLOW)⚠️ No GPU detected$(RESET)"; \
	fi
	@echo ""
	@if [ -d "$(ROOT_DIR)/.venv" ]; then \
		echo "$(BOLD)$(GREEN)📦 Virtual Environment:$(RESET)"; \
		echo "  $(BOLD)Status:$(RESET)           $(GREEN)✅ Active$(RESET)"; \
		echo "  $(BOLD)Packages:$(RESET)         $$(cd $(ROOT_DIR) && uv pip list 2>/dev/null | wc -l) installed"; \
		echo ""; \
		echo "$(BOLD)📦 Installed Packages (top 20):$(RESET)"; \
		cd $(ROOT_DIR) && uv pip list 2>/dev/null | head -20 | sed 's/^/  /'; \
		if [ $$(cd $(ROOT_DIR) && uv pip list 2>/dev/null | wc -l) -gt 20 ]; then \
			echo "  $(DIM)... and $$(($$(cd $(ROOT_DIR) && uv pip list 2>/dev/null | wc -l) - 20)) more$(RESET)"; \
		fi; \
		echo ""; \
		echo "$(BOLD)🔍 Key Packages:$(RESET)"; \
		echo -n "  $(BOLD)PyTorch:$(RESET)         "; \
		cd $(ROOT_DIR) && uv run python -c "import torch; print(torch.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)CUDA Available:$(RESET)  "; \
		cd $(ROOT_DIR) && uv run python -c "import torch; print('$(GREEN)✅ Yes$(RESET)' if torch.cuda.is_available() else '$(YELLOW)⚠️ No$(RESET)')" 2>/dev/null || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)OpenCV:$(RESET)          "; \
		cd $(ROOT_DIR) && uv run python -c "import cv2; print(cv2.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)NumPy:$(RESET)           "; \
		cd $(ROOT_DIR) && uv run python -c "import numpy; print(numpy.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)Pandas:$(RESET)          "; \
		cd $(ROOT_DIR) && uv run python -c "import pandas; print(pandas.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)ruff:$(RESET)            "; \
		cd $(ROOT_DIR) && uv run ruff --version 2>/dev/null | head -1 | awk '{print $$2}' && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)black:$(RESET)           "; \
		cd $(ROOT_DIR) && uv run python -c "import black; print(black.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)isort:$(RESET)           "; \
		cd $(ROOT_DIR) && uv run python -c "import isort; print(isort.__version__)" 2>/dev/null && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)mypy:$(RESET)            "; \
		cd $(ROOT_DIR) && uv run mypy --version 2>/dev/null | head -1 | awk '{print $$2}' && echo "" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)🔒 Security Tools Status:$(RESET)"
	@if [ -d "$(ROOT_DIR)/.venv" ]; then \
		echo -n "  $(BOLD)pip-audit:$(RESET)       "; \
		cd $(ROOT_DIR) && uv run pip show pip-audit >/dev/null 2>&1 && echo "$(GREEN)✅ Installed$(RESET)" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)cyclonedx-bom:$(RESET)   "; \
		cd $(ROOT_DIR) && uv run pip show cyclonedx-bom >/dev/null 2>&1 && echo "$(GREEN)✅ Installed$(RESET)" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
		echo -n "  $(BOLD)safety:$(RESET)          "; \
		cd $(ROOT_DIR) && uv run pip show safety >/dev/null 2>&1 && echo "$(GREEN)✅ Installed$(RESET)" || echo "$(YELLOW)⚠️ Not installed$(RESET)"; \
	else \
		echo "  $(YELLOW)⚠️ Virtual environment not found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)📊 Security Reports:$(RESET)"
	@if [ -d "$(SECURITY_DIR)" ]; then \
		echo "  $(BOLD)Directory:$(RESET)        $(CYAN)$(SECURITY_DIR)$(RESET)"; \
		echo -n "  $(BOLD)Files:$(RESET)           "; \
		find $(SECURITY_DIR) -type f 2>/dev/null | wc -l | xargs echo; \
		if [ -f "$(SECURITY_DIR)/audit-report.json" ]; then \
			echo -n "  $(BOLD)audit-report.json:$(RESET)  "; \
			ls -lh $(SECURITY_DIR)/audit-report.json 2>/dev/null | awk '{print $$5}' | xargs -I {} echo "$(CYAN){} $(RESET)"; \
		fi; \
		if [ -f "$(SECURITY_DIR)/sbom-cyclonedx.json" ]; then \
			echo -n "  $(BOLD)sbom-cyclonedx.json:$(RESET) "; \
			ls -lh $(SECURITY_DIR)/sbom-cyclonedx.json 2>/dev/null | awk '{print $$5}' | xargs -I {} echo "$(CYAN){} $(RESET)"; \
		fi; \
		if [ -f "$(SECURITY_DIR)/safety-report.json" ]; then \
			echo -n "  $(BOLD)safety-report.json:$(RESET)  "; \
			ls -lh $(SECURITY_DIR)/safety-report.json 2>/dev/null | awk '{print $$5}' | xargs -I {} echo "$(CYAN){} $(RESET)"; \
		fi; \
	else \
		echo "  $(YELLOW)⚠️ Security directory not found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)📝 Logs:$(RESET)"
	@if [ -d "$(LOG_DIR)" ]; then \
		echo "  $(BOLD)Directory:$(RESET)        $(CYAN)$(LOG_DIR)$(RESET)"; \
		echo -n "  $(BOLD)Files:$(RESET)           "; \
		find $(LOG_DIR) -type f 2>/dev/null | wc -l | xargs echo; \
		echo "  $(BOLD)Recent logs:$(RESET)"; \
		ls -lt $(LOG_DIR)/*.log 2>/dev/null | head -5 | awk '{print "    " $$9 " (" $$5 ")"}' || echo "    $(DIM)No logs found$(RESET)"; \
	else \
		echo "  $(YELLOW)⚠️ Log directory not found$(RESET)"; \
	fi
	@echo ""
	@echo "$(BOLD)$(GREEN)💡 Quick Commands:$(RESET)"
	@echo "  $(DIM)make setup              # Install dependencies$(RESET)"
	@echo "  $(DIM)make check-code         # Check code style$(RESET)"
	@echo "  $(DIM)make format-all         # Format all code$(RESET)"
	@echo "  $(DIM)make security-all       # Run all security checks$(RESET)"
	@echo "  $(DIM)uv run python -m src.main --dataset tid2013 --model resnet_iqa  # Train$(RESET)"
	@echo ""