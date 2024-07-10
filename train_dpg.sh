export OUTPUT_DIR="db_backpack"
CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file default_config.yaml train_dreambooth_dpg.py \
 --pretrained_model_name_or_path ../../../models/stable-diffusion-v1-4 \
  --instance_data_dir ../../../datasets/dreambooth_datasets/dataset/backpack_dog \
  --instance_prompt "a photo of sks backpack" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir="path_class_images_backpack" \
  --output_dir=$OUTPUT_DIR \
  --class_prompt="a photo of backpack" \
  --resolution=512 --train_batch_size=1 --max_train_steps=1000 --learning_rate=1e-6  \
  --num_class_images=8 --lr_warmup_steps=0 \
  --lr_scheduler="constant" \
  --train_text_encoder

# CUDA_VISIBLE_DEVICES=7 python generate_faces.py --ckpt_path $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0 python generate_db_prompts.py --ckpt_path db_/backpack
