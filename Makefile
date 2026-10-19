SHELL := /bin/bash
ENV_NAME ?= flyloop
VENV ?= .venv
FLYVIS_ROOT_DIR ?= $(CURDIR)/data/flyvis

# On the Mac we use a uv venv (.venv); elsewhere (Adroit) the conda env from
# environment.yml, which must be activated before calling make.
ifneq ($(shell command -v uv 2>/dev/null),)
  BIN := $(VENV)/bin/
else
  BIN :=
endif
PY := $(BIN)python

.PHONY: env test

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
