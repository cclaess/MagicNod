cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

python3 -u experiments/inference_instruct_pix2pix.py \
    --model-path "./checkpoints/instruct_pix2pix_2/" \
    --prompt "Put a benign lung nodule in the masked region" \
    --data-dir "./results/LIDC-IDRI/inpainting/" \
    --output-dir "./results/LIDC-IDRI/pix2pix-benign-increased-guidance-2" \
    --guidance-scale 50.0 \
