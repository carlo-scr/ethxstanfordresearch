CORE_PYTHON ?= python3
RESEARCH_PYTHON ?= python

.PHONY: check test lint smoke validate notebooks-build notebooks-execute notebooks-check notebooks-refresh paper clean-paper

check: test validate

test:
	$(CORE_PYTHON) -m unittest discover -s tests -v

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

smoke:
	PYTHONPATH=src $(CORE_PYTHON) -m latent_safety.cli pilot-smoke --config configs/pilot/smoke.toml --output /tmp/latent-safety-smoke.json

validate:
	$(CORE_PYTHON) scripts/validate_claims.py
	$(CORE_PYTHON) scripts/validate_configs.py
	$(CORE_PYTHON) scripts/validate_notebooks.py --require-executed

notebooks-build:
	$(RESEARCH_PYTHON) scripts/build_notebooks.py

notebooks-execute:
	$(RESEARCH_PYTHON) scripts/execute_notebooks.py

notebooks-check:
	$(RESEARCH_PYTHON) scripts/validate_notebooks.py --require-executed

notebooks-refresh: notebooks-build notebooks-execute notebooks-check

paper:
	$(MAKE) -C paper

clean-paper:
	$(MAKE) -C paper clean
