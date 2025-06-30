# Makefile for Family Contact Agent
# A passion project to foster love and communication within families

.PHONY: help install install-dev test test-verbose test-coverage lint format clean setup whatsapp-bridge download-models run check-deps

# Default target
help:
	@echo "Family Contact Agent - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          - Set up the development environment"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  check-deps     - Check if required system dependencies are installed"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests with pytest"
	@echo "  test-verbose   - Run tests with verbose output"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with Black and isort"
	@echo ""
	@echo "Development:"
	@echo "  whatsapp-bridge - Start WhatsApp MCP bridge"
	@echo "  download-models - Download required model files"
	@echo "  run            - Run the main application"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean up generated files and caches"

# Setup development environment
setup: check-deps
	@echo "Setting up development environment using uv..."
	uv sync
	@echo "Virtual environment created and dependencies installed. Activate it with:"
	@echo "  source .venv/bin/activate  # On Unix/macOS"
	@echo "  .venv\\Scripts\\activate     # On Windows"
	@echo ""
	@echo "Development environment is ready."

# Check system dependencies
check-deps:
	@echo "Checking system dependencies..."
	@echo "Checking ffmpeg..."
	@if command -v ffmpeg >/dev/null 2>&1; then \
		echo "✓ ffmpeg is installed: $(ffmpeg -version | head -n1)"; \
	else \
		echo "✗ ffmpeg is not installed"; \
		echo "  Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Ubuntu)"; \
		exit 1; \
	fi
	@echo "Checking Go..."
	@if command -v go >/dev/null 2>&1; then \
		echo "✓ Go is installed: $(go version)"; \
	else \
		echo "✗ Go is not installed"; \
		echo "  Install from: https://golang.org/dl/"; \
		exit 1; \
	fi
	@echo "Checking uv..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "✓ uv is installed: $(uv --version)"; \
	else \
		echo "✗ uv is not installed"; \
		echo "  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@echo "All system dependencies are installed! ✓"

# Install production dependencies
install:
	@echo "Installing production dependencies..."
	uv pip install -e .

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	uv pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Run tests
test:
	@echo "Running tests..."
	pytest . -v

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	pytest . -vv

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest . --cov=translation_agent --cov-report=term-missing --cov-report=html

# Run formatting and linting using pre-commit
format lint:
	@echo "Running formatting and linting checks with pre-commit..."
	pre-commit run --all-files


# Start WhatsApp MCP bridge
whatsapp-bridge: check-deps
	@echo "Starting WhatsApp MCP bridge..."
	cd whatsapp_mcp/whatsapp-bridge && go run main.go

# Clean up generated files and caches
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleanup complete!"
