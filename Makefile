.PHONY: install smoke test verify

install:
	python -m pip install -e .

smoke:
	bash run_smoke.sh

test:
	PYTHONPATH=code pytest -q tests/test_core.py

verify:
	python scripts/verify_public_release.py
