export WANDB_API_KEY="9c9ca3cd0f56b3e1ec268da8b0b8d545fbe85946"
export WANDB_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_CONFIG_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_CACHE_DIR="/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/"
export WANDB_START_METHOD="thread"
wandb login

export PYTHONPATH="${PYTHONPATH}:/gpfs/work4/0/tese0618/Projects/MagicNod"

cd "/gpfs/work4/0/tese0618/Projects/MagicNod/experiments" || exit

torchrun --nproc_per_node=4 train_inpainting_unet_dist.py \
    --data_dir="/gpfs/work4/0/tese0618/Datasets/LIDC-IDRI-Processed-GenAI" \
    --out_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/models/InpaintingUNet-dist" \
    --experiment="test_training" \
    --batch_size=320 \
    --lr=0.0001 \
    --epochs=10 \
    --loss="ssim" \
    --optimizer="adamw" \
    --scheduler="cosine" \
    --val_split=0.2 \
    --seed=42