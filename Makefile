# OpenEval Lab Makefile
# Provides convenient shortcuts for common tasks

.PHONY: help install test lint format eval benchmark clean docs
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
OPENEVAL := python -m openeval

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)OpenEval Lab - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package and dependencies
	@echo "$(BLUE)Installing OpenEval Lab...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install with development dependencies
	@echo "$(BLUE)Installing OpenEval Lab with dev dependencies...$(NC)"
	$(PIP) install -e '.[dev,openai,hf,metrics]'
	@echo "$(GREEN)✓ Development installation complete$(NC)"

test: ## Run test suite
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --cov=src/openeval --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-fast: ## Run fast tests only (skip slow integration tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -m "not slow"
	@echo "$(GREEN)✓ Fast tests complete$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting...$(NC)"
	ruff check src/ tests/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	ruff check --fix src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/openeval --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Evaluation shortcuts
eval: ## Run quick evaluation demo
	@echo "$(BLUE)Running quick evaluation demo...$(NC)"
	$(OPENEVAL) run examples/qa_spec.json --records --artifacts artifacts/demo
	@echo "$(GREEN)✓ Demo evaluation complete$(NC)"
	@echo "$(YELLOW)Results saved to: artifacts/demo/$(NC)"

eval-qa: ## Run QA benchmark
	@echo "$(BLUE)Running QA benchmark...$(NC)"
	$(OPENEVAL) run examples/qa_spec.json --records --artifacts artifacts/qa --statistical
	@echo "$(GREEN)✓ QA benchmark complete$(NC)"

eval-code: ## Run code generation benchmark
	@echo "$(BLUE)Running code generation benchmark...$(NC)"
	$(OPENEVAL) run examples/code_spec.json --records --artifacts artifacts/code
	@echo "$(GREEN)✓ Code benchmark complete$(NC)"

eval-multimodal: ## Run multimodal evaluation
	@echo "$(BLUE)Running multimodal evaluation...$(NC)"
	$(OPENEVAL) run examples/multimodal_spec.json --records --artifacts artifacts/multimodal
	@echo "$(GREEN)✓ Multimodal evaluation complete$(NC)"

benchmark: ## Run comprehensive benchmark suite
	@echo "$(BLUE)Running comprehensive benchmark suite...$(NC)"
	@mkdir -p artifacts/benchmarks
	$(OPENEVAL) run examples/qa_spec.json --records --artifacts artifacts/benchmarks/qa --statistical
	$(OPENEVAL) run examples/sum_spec.json --records --artifacts artifacts/benchmarks/sum
	@if [ -f examples/code_spec.json ]; then \
		$(OPENEVAL) run examples/code_spec.json --records --artifacts artifacts/benchmarks/code; \
	fi
	$(OPENEVAL) runs collect --dir artifacts/benchmarks --out artifacts/benchmarks/index.json
	@echo "$(GREEN)✓ Benchmark suite complete$(NC)"
	@echo "$(YELLOW)Results aggregated in: artifacts/benchmarks/index.json$(NC)"

benchmark-fast: ## Run fast benchmark subset
	@echo "$(BLUE)Running fast benchmarks...$(NC)"
	@mkdir -p artifacts/fast-benchmarks
	$(OPENEVAL) run examples/qa_spec.json --limit 50 --artifacts artifacts/fast-benchmarks/qa
	$(OPENEVAL) run examples/sum_spec.json --limit 20 --artifacts artifacts/fast-benchmarks/sum
	@echo "$(GREEN)✓ Fast benchmarks complete$(NC)"

# Validation and quality checks
validate: ## Validate all example specs and configurations
	@echo "🔍 Validating specs and configurations..."
	@for spec in examples/*.json examples/*.yaml; do
		if [ -f "$$spec" ]; then
			echo "Validating $$spec...";
			python -m openeval validate-comprehensive "$$spec" --strict || exit 1;
		fi;
	done
	@echo "✅ All validations passed!"

validate-comprehensive: ## Run comprehensive validation on all project files
	@echo "🔍 Running comprehensive validation..."
	@find examples -name "*.json" -o -name "*.yaml" | head -10 | while read file; do
		echo "Validating $$file...";
		python -m openeval validate-comprehensive "$$file" --check-imports --check-datasets --check-performance;
	done
	@echo "✅ Comprehensive validation complete!"

validate-examples: ## Validate all example files quickly
	@echo "🔍 Quick validation of examples..."
	@python scripts/validate_examples.py
	@echo "✅ Example validation complete!"

validate-configs: ## Validate configuration files
	@echo "🔍 Validating configuration files..."
	@find . -name "*.yaml" -path "./configs/*" | while read config; do
		echo "Validating config $$config...";
		python -m openeval validate-comprehensive "$$config" --type config;
	done
	@echo "✅ Configuration validation complete!"

doctor: ## Run system diagnostics
	@echo "$(BLUE)Running system diagnostics...$(NC)"
	$(OPENEVAL) doctor
	@echo "$(GREEN)✓ Diagnostics complete$(NC)"

# Development tasks
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/html; \
		echo "$(GREEN)✓ Documentation generated$(NC)"; \
		echo "$(YELLOW)Open docs/_build/html/index.html$(NC)"; \
	else \
		echo "$(YELLOW)Sphinx not installed. Install with: pip install sphinx$(NC)"; \
	fi

serve: ## Start development web server
	@echo "$(BLUE)Starting web dashboard...$(NC)"
	$(OPENEVAL) web --reload
	@echo "$(GREEN)✓ Web server started at http://localhost:8000$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf artifacts/demo/ artifacts/benchmarks/ artifacts/fast-benchmarks/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Performance testing
perf: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	@mkdir -p artifacts/perf
	time $(OPENEVAL) run examples/qa_spec.json --limit 100 --artifacts artifacts/perf/qa-100
	time $(OPENEVAL) run examples/qa_spec.json --limit 500 --artifacts artifacts/perf/qa-500
	@echo "$(GREEN)✓ Performance benchmarks complete$(NC)"

memory: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(NC)"
	@if command -v mprof >/dev/null 2>&1; then \
		mprof run $(OPENEVAL) run examples/qa_spec.json --limit 100; \
		mprof plot --output=artifacts/memory-profile.png; \
		echo "$(GREEN)✓ Memory profile saved to artifacts/memory-profile.png$(NC)"; \
	else \
		echo "$(YELLOW)memory_profiler not installed. Install with: pip install memory_profiler$(NC)"; \
	fi

# Release tasks
version: ## Show current version
	@$(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Packages built in dist/$(NC)"

# Quick start sequence
quickstart: install validate eval ## Complete quickstart sequence
	@echo ""
	@echo "$(GREEN)🎉 OpenEval Lab quickstart complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  • View results: $(YELLOW)openeval web$(NC)"
	@echo "  • Run benchmarks: $(YELLOW)make benchmark$(NC)"
	@echo "  • Read docs: $(YELLOW)open README.md$(NC)"
	@echo ""

# CI shortcuts
ci: lint type-check test ## Run all CI checks
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

ci-fast: lint test-fast ## Run fast CI checks
	@echo "$(GREEN)✓ Fast CI checks passed$(NC)"

# Development workflow
dev-setup: install-dev ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"; \
	else \
		echo "$(YELLOW)pre-commit not available, skipping hooks$(NC)"; \
	fi
	@echo "$(GREEN)✓ Development environment ready$(NC)"

# Examples for README
examples: ## Generate example outputs for documentation
	@echo "$(BLUE)Generating example outputs...$(NC)"
	@mkdir -p artifacts/examples
	$(OPENEVAL) run examples/qa_spec.json --limit 5 --records --artifacts artifacts/examples/qa
	$(OPENEVAL) runs collect --dir artifacts/examples --out artifacts/examples/index.json
	@echo "$(GREEN)✓ Example outputs generated$(NC)"

# Show project status
status: ## Show project status and metrics
	@echo "$(BLUE)OpenEval Lab Project Status$(NC)"
	@echo ""
	@echo "$(YELLOW)Version:$(NC) $$(make version)"
	@echo "$(YELLOW)Examples:$(NC) $$(ls examples/*.json | wc -l) specifications"
	@echo "$(YELLOW)Tests:$(NC) $$(find tests -name "test_*.py" | wc -l) test files"
	@echo "$(YELLOW)Coverage:$(NC) Run 'make test' to see coverage"
	@echo ""
	@if [ -d .git ]; then \
		echo "$(YELLOW)Git Status:$(NC)"; \
		git status --short; \
		echo "$(YELLOW)Recent Commits:$(NC)"; \
		git log --oneline -5; \
	fi

# Health and monitoring
health:				## Run project health dashboard
	@python scripts/project_health.py

health-json:			## Run project health dashboard (JSON output)
	@python scripts/project_health.py --json

health-report:			## Generate detailed health report
	@python scripts/project_health.py --save project-health-report.json
