#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=0:30:00


mkdir -p /gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/
cd /gpfs/work4/0/tese0618/Projects/MagicNod/scripts/

srun apptainer exec --nv /gpfs/work4/0/tese0618/Containers/misc_v6.sif /bin/bash train_instructpix2pix.sh