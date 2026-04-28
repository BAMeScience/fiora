.PHONY: all download preprocess

PYTHON ?= python3
DOWNLOAD_ARGS ?=
PREPROCESS_ARGS ?=

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DOWNLOAD_SCRIPT := $(SCRIPT_DIR)download_msnlib.py
PREPROCESS_SCRIPT := $(SCRIPT_DIR)preprocess_msnlib.py

all: download preprocess

download:
	$(PYTHON) $(DOWNLOAD_SCRIPT) $(DOWNLOAD_ARGS)

preprocess:
	$(PYTHON) $(PREPROCESS_SCRIPT) $(PREPROCESS_ARGS)
