.PHONY: setup setup-modal corpus benchmark replicated-benchmark shape-sweep modal-replicated modal-shape-sweep modal-white-nights artifacts test
PYTHON ?= .venv/bin/python
MODAL ?= .venv/bin/modal

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

setup-modal:
	$(PYTHON) -m pip install -r requirements-modal.txt
	$(MODAL) setup

corpus:
	$(PYTHON) scripts/fetch_white_nights.py

benchmark:
	PYTHONPATH=src $(PYTHON) -m vwt_bench.benchmark \
		--steps 500 \
		--eval-iters 16 \
		--eval-interval 100 \
		--history-interval 5 \
		--layers 6 \
		--width 96 \
		--heads 4 \
		--batch-size 32 \
		--block-size 96 \
		--generate-tokens 240

replicated-benchmark:
	PYTHONPATH=src $(PYTHON) -m vwt_bench.benchmark \
		--steps 500 \
		--eval-iters 16 \
		--eval-interval 100 \
		--history-interval 5 \
		--seeds 1337,2027,3141 \
		--layers 6 \
		--width 96 \
		--heads 4 \
		--batch-size 32 \
		--block-size 96 \
		--generate-tokens 240 \
		--report-path runs/replicated_benchmark.json

shape-sweep:
	PYTHONPATH=src $(PYTHON) -m vwt_bench.benchmark \
		--steps 500 \
		--eval-iters 16 \
		--eval-interval 100 \
		--history-interval 5 \
		--variable-shapes x,diamond,increasing,decreasing \
		--layers 6 \
		--width 96 \
		--heads 4 \
		--batch-size 32 \
		--block-size 96 \
		--generate-tokens 240 \
		--report-path runs/shape_sweep.json

modal-replicated:
	$(MODAL) run scripts/modal_benchmark.py --mode replicated

modal-shape-sweep:
	$(MODAL) run scripts/modal_benchmark.py --mode shape-sweep

modal-white-nights:
	$(MODAL) run scripts/modal_benchmark.py --mode white-nights-replicated

artifacts:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py --report runs/last_run.json --out-dir runs/artifacts

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
