cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

python3 -u experiments/nodule_synthesis.py \
    --model-path "./checkpoints/instruct_pix2pix_2/" \
    --prompt "Put a malignant lung nodule in the masked region" \
    --data-dir "./data/LNDb" \
    --num-inference-steps 50 \
    --image-guidance-scale 2.5 \
    --guidance-scale 7.5 \
    --output-dir "./results/LNDb/nodule_synthesis" \
