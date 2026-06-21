.PHONY: setup quick artifacts test
PYTHON ?= .venv/bin/python

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

quick:
	PYTHONPATH=src $(PYTHON) -m vwt_bench.benchmark --steps 80 --eval-iters 8 --batch-size 16

artifacts:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py --report runs/last_run.json --out-dir runs/artifacts

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
