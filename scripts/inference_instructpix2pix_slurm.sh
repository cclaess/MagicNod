#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

cd "/gpfs/work4/0/tese0618/Projects/MagicNod"

srun apptainer exec --nv "/gpfs/work4/0/tese0618/Containers/magicnod_v2.sif" \
        python3 -u experiments/inference_instructpix2pix.py \
        --model_path "/gpfs/work4/0/tese0618/Projects/MagicNod/models/InstructPix2Pix" \
        --prompt "Put a lung nodule in the red rectangles" \
        --data_path "/gpfs/work4/0/tese0618/Datasets/LNDb-Processed-GenAI" \
        --output_path "/gpfs/work4/0/tese0618/Projects/MagicNod/output/InstructPix2Pix/LNDb"