cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

python3 -u experiments/nodule_synthesis.py \
    --model-path-vqvae "./checkpoints/test_run_lidc_7/best_model.pth" \
    --model-path-pix2pix "./checkpoints/instruct_pix2pix_2/" \
    --data-dir "./data/LNDb" \
    --output-dir "./results/LNDb/nodule_synthesis" \
