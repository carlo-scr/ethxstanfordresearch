.PHONY: check test lint smoke validate notebooks-build notebooks-execute notebooks-check paper clean-paper

check: test validate

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

smoke:
	PYTHONPATH=src python3 -m latent_safety.cli pilot-smoke --config configs/pilot/smoke.toml --output /tmp/latent-safety-smoke.json

validate:
	python3 scripts/validate_claims.py
	python3 scripts/validate_configs.py
	python3 scripts/validate_notebooks.py --require-executed

notebooks-build:
	python scripts/build_notebooks.py

notebooks-execute:
	python scripts/execute_notebooks.py

notebooks-check:
	python3 scripts/validate_notebooks.py --require-executed

paper:
	$(MAKE) -C paper

clean-paper:
	$(MAKE) -C paper clean
