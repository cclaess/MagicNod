cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

accelerate launch --multi-gpu --mixed_precision "fp16" \
    experiments/train_inpainting_unet.py \
    --experiment-name "test_run_lidc" \
    --data-dir "./data/LIDC-IDRI" \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.00001 \
    --num-workers 10 \
    --wandb \
    --seed 42 \
