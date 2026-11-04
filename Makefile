.PHONY: help install install-dev lint format type test cov audit clean docker-up docker-down

UV      ?= uv
PY      ?= $(UV) run python
PYTEST  ?= $(UV) run pytest
RUFF    ?= $(UV) run ruff
MYPY    ?= $(UV) run mypy

help:
	@echo "Targets:"
	@echo "  install       Install production deps (uv sync --no-dev)"
	@echo "  install-dev   Install all deps + pre-commit hooks"
	@echo "  lint          ruff check"
	@echo "  format        ruff format + ruff check --fix"
	@echo "  type          mypy (strict) on src/vnfake"
	@echo "  test          unit tests (excl. slow/integration)"
	@echo "  cov           unit tests with coverage"
	@echo "  audit         Run Phase 0 data audit script"
	@echo "  docker-up     Start Qdrant + MLflow"
	@echo "  docker-down   Stop docker compose"
	@echo "  clean         Remove caches"

install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync --all-extras
	$(UV) run pre-commit install

lint:
	$(RUFF) check src scripts tests

format:
	$(RUFF) format src scripts tests
	$(RUFF) check --fix src scripts tests

type:
	$(MYPY) src/vnfake

test:
	$(PYTEST) -m "not slow and not integration" tests/unit

cov:
	$(PYTEST) -m "not slow and not integration" \
		--cov=src/vnfake --cov-report=term-missing --cov-report=html tests/

audit:
	$(PY) scripts/01_audit_data.py

audit-overlap:
	$(PY) scripts/01b_verify_overlap.py --split dev

audit-leakage:
	$(PY) scripts/01c_check_leakage.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov/

prepare-data:
	$(PY) scripts/02_prepare_data.py

prepare-data-fast:
	$(PY) scripts/02_prepare_data.py --skip-segmentation

index-kb:
	$(PY) scripts/03_index_kb.py

index-kb-force:
	$(PY) scripts/03_index_kb.py --force
