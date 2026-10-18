.PHONY: help install setup clean run index inspect test lint format

# Default Python interpreter
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

help:
	@echo "YouTube RAG - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install dependencies only"
	@echo "  make env            - Copy .env.example to .env"
	@echo ""
	@echo "Running:"
	@echo "  make run            - Start the Gradio web interface"
	@echo "  make index [CHANNEL=@name] [MAX_VIDEOS=N] - Index a YouTube channel"
	@echo "  make inspect        - Inspect indexed videos and channels"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove temporary files and caches"
	@echo "  make clean-all      - Remove everything including venv and data"
	@echo ""

setup: $(VENV)/bin/activate env
	@echo "✓ Setup complete! Activate venv with: source $(BIN)/activate"

$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .
	@echo "✓ Virtual environment created and dependencies installed"

activate:
	@echo "To activate the virtual environment, run: source $(BIN)/activate"

install:
	@echo "Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .
	@echo "✓ Dependencies installed"

env:
	@if [ ! -f .env ]; then \
		echo "Creating .env file from .env.example..."; \
		cp .env.example .env; \
		echo "✓ .env file created. Please edit it with your API keys."; \
	else \
		echo ".env file already exists"; \
	fi

run:
	@echo "Starting Gradio app..."
	$(BIN)/python -m src.app

index:
	@echo "Running indexing script..."
	$(BIN)/python scripts/test_indexer.py $(if $(CHANNEL),$(CHANNEL),) $(if $(MAX_VIDEOS),--max-videos $(MAX_VIDEOS),)

inspect:
	@echo "Inspecting ChromaDB index..."
	$(BIN)/python scripts/inspect_index.py

lint:
	@echo "Running linting checks..."
	@if command -v ruff >/dev/null 2>&1; then \
		$(BIN)/ruff check src/ tests/; \
	else \
		echo "ruff not installed. Install with: pip install ruff"; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		$(BIN)/black src/ tests/ scripts/; \
		echo "✓ Code formatted"; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo "✓ Cleaned temporary files"

clean-data:
	@echo "Removing data in data/chroma_db/ and data/cache/..."
	rm -rf data/chroma_db/*
	rm -rf data/cache/*
	@echo "✓ Data cleanup done"

clean-all: clean
	@echo "Removing virtual environment and data..."
	rm -rf $(VENV)
	rm -rf data/chroma_db/*
	rm -rf data/cache/*
	@echo "✓ Complete cleanup done"
