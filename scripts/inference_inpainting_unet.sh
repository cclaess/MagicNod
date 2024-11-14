cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

python3 experiments/inference_inpainting_unet.py \
    --model-path "./checkpoints/test_run_lidc_2/best_model.pth" \
    --data-dir "./data/LIDC-IDRI" \
    --output-dir "./results/LIDC-IDRI/inpainting" \
