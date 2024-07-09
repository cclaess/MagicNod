export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR=/gpfs/work4/0/tese0618/Projects/MagicNod/models/InstructPix2Pix/
export DATASET_PATH="/gpfs/work4/0/tese0618/Projects/MagicNod/dataset/"

export WANDB_API_KEY="9c9ca3cd0f56b3e1ec268da8b0b8d545fbe85946"
export WANDB_DIR=/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/
export WANDB_CONFIG_DIR=/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/
export WANDB_CACHE_DIR=/gpfs/work4/0/tese0618/Projects/MagicNod/wandb/$SLURM_JOBID/
export WANDB_START_METHOD="thread"
wandb login

cd "/gpfs/work4/0/tese0618/Projects/MagicNod/generativezoo/models/SD"

accelerate launch --mixed_precision="fp16" InstructPix2Pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_PATH \
    --output_dir=$OUTPUT_DIR \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --validation_image=/gpfs/work4/0/tese0618/Projects/MagicNod/dataset/original_images/LIDC-IDRI-0001_01-01-2000-NA-NA-30178_3000566.000000-NA-03192_90.png \
    --validation_prompt="fill the red squares with pulmonary nodules" \
    --seed=42 \
    --report_to="wandb" \
    --original_image_colum="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --dataloader_num_workers=10