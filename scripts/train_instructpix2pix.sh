cd ~/Projects/MagicNod | exit 1

mkdir -p $WANDB_DIR
wandb login

accelerate launch --mixed_precision="fp16" generativezoo/models/SD/InstructPix2Pix.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
    --train_data_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/results/LIDC-IDRI/inpainting/" \
    --output_dir="/gpfs/work4/0/tese0618/Projects/MagicNod/checkpoints/instruct_pix2pix_1" \
    --resolution=512 --random_flip \
    --train_batch_size=32 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=25000 \
    --checkpointing_steps=5000 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --validation_image="/gpfs/work4/0/tese0618/Projects/MagicNod/results/LIDC-IDRI/inpainting/valid/LIDC-IDRI-0007/01-01-2000-NA-NA-81781/3000631.000000-NA-57680/combined_mask_slice=0007_nod=1.png" \
    --validation_prompt="Put a malignant pulmonary nodule in the masked region." \
    --seed=42 \
    --report_to="wandb" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --dataloader_num_workers=10 \