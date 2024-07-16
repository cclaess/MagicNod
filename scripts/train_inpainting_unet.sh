cd "/gpfs/work4/0/tese0618/Projects/MagicNod/experiments" || exit

python3 train_inpainting_unet.py \
    --data_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/Datasets/LIDC-IDRI-Processed-GenAI" \
    --out_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/models/InpaintingUNet"