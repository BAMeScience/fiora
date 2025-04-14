#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=jupyter-%J.oute

conda activate fiora
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0