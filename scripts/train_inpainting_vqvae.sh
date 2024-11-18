cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

accelerate launch --multi-gpu \
    experiments/train_inpainting_vqvae.py \
    --experiment-name "test_run_lidc_1" \
    --data-dir "./data/LIDC-IDRI" \
    --batch-size 4 \
    --epochs 100 \
    --lr-g 1e-7 \
    --lr-d 5e-7 \
    --num-workers 10 \
    --wandb \
    --seed 42 \
