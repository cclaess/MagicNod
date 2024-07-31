#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=6:00:00

cd "/gpfs/work4/0/tese0618/Projects/MagicNod"

srun apptainer exec --nv "/gpfs/work4/0/tese0618/Containers/magicnod_v2.sif" \
        python3 -u experiments/inference_inpainting_unet.py \
        --model "/gpfs/work4/0/tese0618/Projects/MagicNod/models/InpaintingUNet-dist-4/model-epoch=04-val_loss=1.19.ckpt" \
        --input "/gpfs/work4/0/tese0618/Datasets/LIDC-IDRI-Processed-GenAI" \
        --output "/gpfs/work4/0/tese0618/Datasets/LIDC-IDRI-Processed-GenAI"