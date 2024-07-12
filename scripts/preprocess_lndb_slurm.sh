#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

cd "/gpfs/work4/0/tese0618/Projects/MagicNod"

srun apptainer exec --nv "/gpfs/work4/0/tese0618/Containers/misc_v6.sif" python3 scripts/preprocess_lndb.py --input_dir "/gpfs/work4/0/tese0618/Datasets/LNDb-Processed" --output_dir "/gpfs/work4/0/tese0618/Datasets/LNDb-Processed-GenAI"