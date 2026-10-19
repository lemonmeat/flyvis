SHELL := /bin/bash
ENV_NAME ?= flyloop
FLYVIS_ROOT_DIR ?= $(CURDIR)/data/flyvis
RUN := conda run --no-capture-output -n $(ENV_NAME)

.PHONY: env test

env:
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	FLYVIS_ROOT_DIR=$(FLYVIS_ROOT_DIR) $(RUN) flyvis download-pretrained

test:
	$(RUN) pytest -q
