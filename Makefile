SHELL := /bin/bash
ENV_NAME ?= flyloop
VENV ?= .venv
FLYVIS_ROOT_DIR ?= $(CURDIR)/data/flyvis

# Adroit (ssh host alias `adroit`, see README).
REMOTE ?= adroit
REMOTE_USER ?= ag0226
REMOTE_CODE ?= ~/cos597z
REMOTE_ROOT ?= /scratch/network/$(REMOTE_USER)/flyvis-closedloop
REMOTE_RUNS ?= $(REMOTE_ROOT)/runs

# On the Mac we use a uv venv (.venv); elsewhere (Adroit) the conda env from
# environment.yml, which must be activated before calling make.
ifneq ($(shell command -v uv 2>/dev/null),)
  BIN := $(VENV)/bin/
else
  BIN :=
endif
PY := $(BIN)python

.PHONY: env test tuning bench submit pull

env:
ifneq ($(shell command -v uv 2>/dev/null),)
	uv venv --python 3.10 $(VENV)
	uv pip install --python $(VENV)/bin/python -e ".[dev]"
else
	conda env create -f environment.yml || conda env update -f environment.yml --prune
endif
	FLYVIS_ROOT_DIR=$(FLYVIS_ROOT_DIR) $(BIN)flyvis download-pretrained

test:
	$(PY) -m pytest -q

tuning:
	$(PY) scripts/run_openloop_tuning.py $(ARGS)

bench:
	$(PY) scripts/benchmark_speed.py $(ARGS)

# make submit SWEEP=openloop_tuning [ARRAY=0-9] [ARGS="key=value ..."]
# Runs scripts/run_$(SWEEP).py with configs/$(SWEEP).yaml as a Slurm array, one
# index per sweep_list entry (ARRAY defaults to all of them).
submit:
	@test -n "$(SWEEP)" || { echo "usage: make submit SWEEP=<name>"; exit 1; }
	@test -f scripts/run_$(SWEEP).py || { echo "missing scripts/run_$(SWEEP).py"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "commit your changes first (code moves by git only)"; exit 1; }
	$(eval ARRAY ?= 0-$(shell $(PY) -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/$(SWEEP).yaml'); print(max(len(c.get('sweep_list') or []), 1) - 1)"))
	git push
	ssh $(REMOTE) 'cd $(REMOTE_CODE) && git pull --ff-only && mkdir -p $(REMOTE_ROOT)/logs && sbatch --array=$(ARRAY) --job-name=$(SWEEP) slurm/array_template.slurm scripts/run_$(SWEEP).py $(ARGS)'

# Bring back small results only (csv, json, png); raw activity stays on scratch.
pull:
	rsync -avz --prune-empty-dirs --include='*/' --include='*.csv' --include='*.json' \
		--include='*.png' --exclude='*' $(REMOTE):$(REMOTE_RUNS)/ runs/
