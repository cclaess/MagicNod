#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00


mkdir -p "/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
cd "/gpfs/work4/0/tese0618/Projects/MagicNod/scripts/" || exit

srun apptainer exec --nv /gpfs/work4/0/tese0618/Containers/magicnod_v2.sif /bin/bash train_inpainting_unet.sh