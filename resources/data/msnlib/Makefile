.PHONY: all download preprocess bond-energies

PYTHON ?= python3
DOWNLOAD_ARGS ?=
PREPROCESS_ARGS ?=

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DOWNLOAD_SCRIPT := $(SCRIPT_DIR)download_msnlib.py
PREPROCESS_SCRIPT := $(SCRIPT_DIR)preprocess_msnlib.py
BOND_ENERGY_SCRIPT := $(SCRIPT_DIR)preprocess_bond_energies.py
LIBRARY_CSV := $(SCRIPT_DIR)library.csv
LIBRARY_BOND_ENERGY_CSV ?= $(SCRIPT_DIR)library_with_bond_energies.csv
BOND_ENERGY_CHECKPOINT_EVERY ?= 25
BOND_ENERGY_GPU_WORKERS ?= 1
BOND_ENERGY_INFERENCE_MODE ?= relaxed
BOND_ENERGY_OPTIMIZER ?= fire_bfgs
BOND_ENERGY_FMAX ?= 0.05
BOND_ENERGY_MAX_STEPS ?= 200
BOND_ENERGY_BATCH_SIZE ?= 256
BOND_ENERGY_MODEL ?= medium
BOND_ENERGY_MODEL_WRAPPER ?= mace
BOND_ENERGY_MACE_SOURCE ?= off
BOND_ENERGY_EXTRA_ARGS ?=
# Backward-compatible alias used by experiment Makefiles.
PREPROCESS_EXTRA_ARGS ?=
BOND_ENERGY_ARGS ?= \
	--input $(LIBRARY_CSV) \
	--output $(LIBRARY_BOND_ENERGY_CSV) \
	--mace-source $(BOND_ENERGY_MACE_SOURCE) \
	--model $(BOND_ENERGY_MODEL) \
	--model-wrapper $(BOND_ENERGY_MODEL_WRAPPER) \
	--device cuda:0 \
	--inference-mode $(BOND_ENERGY_INFERENCE_MODE) \
	--optimizer $(BOND_ENERGY_OPTIMIZER) \
	--fmax $(BOND_ENERGY_FMAX) \
	--max-steps $(BOND_ENERGY_MAX_STEPS) \
	--batch-size $(BOND_ENERGY_BATCH_SIZE) \
	--progress \
	--verbose \
	--checkpoint-every $(BOND_ENERGY_CHECKPOINT_EVERY) \
	--gpu-workers $(BOND_ENERGY_GPU_WORKERS) \
	$(BOND_ENERGY_EXTRA_ARGS) \
	$(PREPROCESS_EXTRA_ARGS)

all: download preprocess

download:
	$(PYTHON) $(DOWNLOAD_SCRIPT) $(DOWNLOAD_ARGS)

preprocess:
	$(PYTHON) $(PREPROCESS_SCRIPT) $(PREPROCESS_ARGS)

bond-energies:
	$(PYTHON) $(BOND_ENERGY_SCRIPT) $(BOND_ENERGY_ARGS)
