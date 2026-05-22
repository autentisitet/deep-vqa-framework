# --- Project Metadata ---
PROJECT_NAME := Deep-VQA-Framework
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON := uv run --project $(ROOT_DIR)/pyproject.toml python


$(info 📂 Project Root detected as: $(ROOT_DIR))
.PHONY: setup data train check clean help network all

# Make sure the first goal is to help
# 	 When you type `make` directly in the terminal without any arguments,
# it will automatically execute the first target that appears in the file.
help:
	@echo "🛠️  $(PROJECT_NAME) Commands:"
	@echo "  make setup      - Install dependencies, optimize network"
	@echo "  make data       - Prepare datasets"
	@echo "  make train      - Start training in background"
	@echo "  make check      - Check training status (GPU/memory/process)"
	@echo "  make clean      - Clean caches and temp files"
	@echo "  make network    - Optimize network/Jupyter settings"
	@echo "  make all        - Full pipeline (setup + data + train)"
	@echo ""
	@echo "📁 Config file: ../config/base_config.yaml"


# 1. Environment Initialization
setup:
	@echo "🔐 Setting script permissions..."
	@chmod +x $(ROOT_DIR)/scripts/*.sh
	@echo "⚙️  Setting up environment..."
	@cd $(ROOT_DIR)/scripts && bash setup_env.sh --mirror
	@echo "✅ Environment ready"


# 2. Data Preparation
data:
	@echo "📦 Preparing datasets..."
	@cd $(ROOT_DIR)/scripts && bash manage_data.sh
	@echo "✅ Data ready"


# 3. Start training (runs in the background)
# nohup must be followed by a real, standalone executable file/program.
train:
	@if [ ! -f $(ROOT_DIR)/scripts/download_flag ]; then \
		echo "⚠️  Dataset flag not found. Preparing datasets..."; \
		make setup; \
		make data; \
	fi
	@mkdir -p $(ROOT_DIR)/results/scripts_logs
	@echo "🚀 Starting training in background..."
	@cd $(ROOT_DIR) && nohup $(PYTHON) -m src.main > results/scripts_logs/train.log 2>&1 &
	@echo "🔥 Training started. PID: $$!"
	@echo "Monitor with: tail -f $(ROOT_DIR)/train.log"


# 4. Status Audit
check:
	@cd $(ROOT_DIR)/scripts && bash system_check.sh


# 5. Network Optimization
network:
	@cd $(ROOT_DIR)/scripts && bash network_control.sh


# 6. Clear cache
clean:
	@cd $(ROOT_DIR)/scripts && bash cache_clean.sh


# 7. Packaging Results
archive:
	@cd $(ROOT_DIR)/scripts && bash archive_results.sh --all


#8. One-click full process (for initial exploration)
all: setup data train
	@echo "🎉 Full pipeline completed!"


#9. Stop training
stop:
	@echo "🛑 Stopping training..."
	@pkill -f "$(ROOT_DIR)/src/main.py" || echo "No training process found"