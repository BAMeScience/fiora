#!/bin/bash

# Request resources and start an interactive session
srun --gres=gpu:a100 --mem=200G --time=72:00:00 --pty bash

# Activate the conda environment
conda activate fiora

# Start Jupyter Notebook
jupyter notebook --no-browser --port=8888 --ip=127.0.0.1
