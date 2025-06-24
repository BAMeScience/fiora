#!/bin/bash

# Request resources and start an interactive session
srun --gres=shard:1 --qos=interactive --mem=200G --time=72:00:00 --pty bash
# --gres=1 --qos=normal

# Activate the conda environment
conda activate fiora

# Start Jupyter Notebook
jupyter notebook --no-browser --port=8888 --ip=127.0.0.1
