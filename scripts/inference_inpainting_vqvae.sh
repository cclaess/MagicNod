cd ~/Projects/MagicNod | exit 1

python3 -u experiments/inference_inpainting_vqvae.py \
    --model-path "./checkpoints/test_run_lidc_7/best_model.pth" \
    --data-dir "./data/LIDC-IDRI" \
    --output-dir "./results/LIDC-IDRI/inpainting" \
    --batch-size 3 \
    --num-workers 10 \
    --seed 42
