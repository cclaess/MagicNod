export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./../models/InstructPix2Pix/"
export DATASET_PATH="./../../../Datasets/LIDC-IDRI-Processed-GenAI/"

cd "./../generativezoo/models/SD"

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
    --validation_image "./../../../../../Datasets/LIDC-IDRI-Processed-GenAI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/Masked/3000566.000000-NA-03192_90.png" \
    --validation_prompt "fill the red squares with a pulmonary nodule" \
    --seed=42 \
    --report_to=wandb \
    --original_image_colum="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt"