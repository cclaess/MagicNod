#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                    	        # Specify the number of GPUs to use
#SBATCH --time=6:00:00                         # Specify the maximum time the job can run

# Umbrella cluster:
# 6x titanrtx.24gb
# 8x rtx2080ti.11gb
# 3x rtxa4500.20gb
# 2x rtx3090ti.24gb

# SURF Snellius HPC:
# 4x a100.40gb
# 4x h100.96gb

set -a
source ./.env

apptainer exec --nv --bind $MOUNT_PATH --env-file ./.env $CONTAINER_PATH \
    python3 -u scripts/preprocess_lndb.py \
        	--data-dir "/gpfs/work4/0/tese0618/Datasets/LNDb" \
            --output-dir "/gpfs/work4/0/tese0618/Projects/MagicNod/data/LNDb"