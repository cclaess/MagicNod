export WANDB_API_KEY="9c9ca3cd0f56b3e1ec268da8b0b8d545fbe85946"
export WANDB_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_CONFIG_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_CACHE_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_START_METHOD="thread"
wandb login

export PYTHONPATH="${PYTHONPATH}:/gpfs/work4/0/tese0618/Projects/MagicNod"

cd "/gpfs/work4/0/tese0618/Projects/MagicNod/experiments" || exit

torchrun --nproc_per_node=1 train_inpainting_unet_dist_lightning.py \
    --data_dir="/gpfs/work4/0/tese0618/Datasets/LIDC-IDRI-Processed-GenAI" \
    --output_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/models/InpaintingUNet-dist-4" \
    --experiment="test_training_4" \
    --batch_size=320 \
    --lr=0.0001 \
    --epochs=5 \
    --loss="wmse" \
    --optimizer="adamw" \
    --scheduler="cosine" \
    --val_split=0.1 \
    --seed=42 \
    --gpus=1 \
    --accelerator="cuda"