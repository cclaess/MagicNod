#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00


mkdir -p "/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
cd "/gpfs/work4/0/tese0618/Projects/MagicNod/scripts/" || exit

srun apptainer exec --nv /gpfs/work4/0/tese0618/Containers/magicnod_v2.sif /bin/bash train_inpainting_unet_dist.sh