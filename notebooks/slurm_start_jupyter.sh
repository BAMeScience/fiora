#!/bin/bash

# Request resources and start an interactive session
# srun --gres=shard:1 --qos=interactive --mem=200G --time=72:00:00 --pty bash
srun --gres=gpu:4g.40gb:1 --qos=interactive --time=72:00:00 --pty bash
# --gres=1 --qos=normal

# Activate the conda environment
conda activate fiora

# Start Jupyter Notebook
jupyter notebook --no-browser --port=8888 --ip=127.0.0.1

# Previously used script (commented out for reference)
# #!/bin/bash
# #SBATCH --gpus=1
# #SBATCH --time=02:00:00
# #SBATCH --mem=16G
# #SBATCH --output=jupyter-%J.oute

# conda activate fiora
# jupyter notebook --no-browser --port=8888 --ip=127.0.0.1