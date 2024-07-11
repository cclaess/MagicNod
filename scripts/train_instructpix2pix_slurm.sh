#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=48:00:00


mkdir -p "/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
cd "/gpfs/work4/0/tese0618/Projects/MagicNod/scripts/" || exit

srun apptainer exec --nv /gpfs/work4/0/tese0618/Containers/misc_v6.sif /bin/bash train_instructpix2pix.sh