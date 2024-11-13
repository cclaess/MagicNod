#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=6:00:00

cd "/gpfs/work4/0/tese0618/Projects/MagicNod"

apptainer exec --nv --bind $MOUNT_PATH --env-file ./.env $CONTAINER_PATH \
    /bin/bash scripts/inference_inpainting_unet.sh