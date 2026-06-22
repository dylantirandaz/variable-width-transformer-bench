.PHONY: setup setup-modal setup-scale corpus benchmark replicated-benchmark shape-sweep
.PHONY: modal-replicated modal-shape-sweep modal-white-nights modal-prepare-dclm
.PHONY: modal-append-terminal-token modal-paper-scale modal-paper-200m artifacts test
PYTHON ?= .venv/bin/python
MODAL ?= .venv/bin/modal
SCALE ?= dense_200m
MODEL_KIND ?= both
TRAIN_BIN ?= /data/dclm_cl100k_uint32.bin
DCLM_DATASET ?= mlfoundations/dclm-baseline-1.0
OVERWRITE ?= false
CHECKPOINT_INTERVAL ?= 1000
CHECKPOINT_AT_END ?= true
KEEP_CHECKPOINTS ?= 2
RESUME ?= true

DCLM_PREP_FLAGS = \
	--scale $(SCALE) \
	--dataset $(DCLM_DATASET)
MODAL_PAPER_FLAGS = \
	--scale $(SCALE) \
	--model-kind $(MODEL_KIND) \
	--train-bin $(TRAIN_BIN) \
	--checkpoint-interval $(CHECKPOINT_INTERVAL) \
	--keep-checkpoints $(KEEP_CHECKPOINTS)

ifneq ($(filter true 1 yes,$(OVERWRITE)),)
DCLM_PREP_FLAGS += --overwrite
endif

ifneq ($(filter false 0 no,$(CHECKPOINT_AT_END)),)
MODAL_PAPER_FLAGS += --no-checkpoint-at-end
endif

ifneq ($(filter false 0 no,$(RESUME)),)
MODAL_PAPER_FLAGS += --no-resume
endif

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

setup-modal:
	$(PYTHON) -m pip install -r requirements-modal.txt
	$(MODAL) setup

setup-scale:
	$(PYTHON) -m pip install -r requirements-scale.txt

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

modal-prepare-dclm:
	$(MODAL) run scripts/modal_prepare_dclm.py $(DCLM_PREP_FLAGS)

modal-append-terminal-token:
	$(MODAL) run scripts/modal_append_terminal_token.py \
		--scale $(SCALE) \
		--output $(TRAIN_BIN)

modal-paper-scale:
	$(MODAL) run scripts/modal_paper_scale.py $(MODAL_PAPER_FLAGS)

modal-paper-200m:
	$(MAKE) modal-paper-scale SCALE=dense_200m MODEL_KIND=both

artifacts:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py --report runs/last_run.json --out-dir runs/artifacts

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
