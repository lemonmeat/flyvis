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

.PHONY: env test tuning bench submit pull status logs

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

# make submit SWEEP=<name> [ARRAY=0-9] [ARGS="key=value ..."] [SBATCH_ARGS="..."]
# Submits scripts/run_<name>.py (or scripts/<name>.py) as a Slurm array through
# slurm/array_template.slurm. With configs/<name>.yaml, ARRAY defaults to one index
# per sweep_list entry; otherwise to a single job (0-0). SBATCH_ARGS passes extra
# sbatch options, e.g. SBATCH_ARGS="--gres=gpu:nvidia_a100:1 --time=00:20:00".
SCRIPT ?= $(firstword $(wildcard scripts/run_$(SWEEP).py scripts/$(SWEEP).py))
submit:
	@test -n "$(SWEEP)" || { echo "usage: make submit SWEEP=<name>"; exit 1; }
	@test -n "$(SCRIPT)" || { echo "no scripts/run_$(SWEEP).py or scripts/$(SWEEP).py"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "commit your changes first (code moves by git only)"; exit 1; }
	$(eval ARRAY ?= $(if $(wildcard configs/$(SWEEP).yaml),0-$(shell $(PY) -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/$(SWEEP).yaml'); print(max(len(c.get('sweep_list') or []), 1) - 1)"),0-0))
	git push
	ssh $(REMOTE) 'cd $(REMOTE_CODE) && git pull --ff-only && mkdir -p $(REMOTE_ROOT)/logs && sbatch --array=$(ARRAY) --job-name=$(SWEEP) $(SBATCH_ARGS) slurm/array_template.slurm $(SCRIPT) $(ARGS)'

# Queue and the last day's jobs, without logging in.
status:
	ssh $(REMOTE) 'squeue -u $(REMOTE_USER) -o "%.12i %.18j %.9T %.10M %.10l %R"; echo; sacct -u $(REMOTE_USER) -S now-1days -X --format=JobID%16,JobName%18,State%11,Elapsed,ExitCode | tail -n 15'

# Tail the newest log (or LOG=<file name> in the logs folder).
logs:
	ssh $(REMOTE) 'cd $(REMOTE_ROOT)/logs && f="$(LOG)" && f=$${f:-$$(ls -t | head -1)} && echo "== $$f" && tail -n 40 "$$f"'

# Bring back small results only (csv, json, png); raw activity stays on scratch.
pull:
	rsync -avz --prune-empty-dirs --include='*/' --include='*.csv' --include='*.json' \
		--include='*.png' --exclude='*' $(REMOTE):$(REMOTE_RUNS)/ runs/
