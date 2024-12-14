## **[ECCV 2024] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning** 
<div align="center">

 <a href='https://arxiv.org/abs/2407.06642v2'><img src='https://img.shields.io/badge/arXiv-2407.06642-b31b1b.svg'></a> &nbsp;
</div>

## 🔆 Introduction

This repo contains the official code of our ECCV2024 paper: [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning]

The ~~paper and~~ code will be release soon in the next 1~2 weeks.

## ⚙️ Setup

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
pip install -r requirements.txt
```

## 💥 Training
### Using 'Look Forward' reward

Take `backpack_dog(backpack)` as example. Put your pretrained model in `path/to/pretrained_stable_diffusion`, We use Stable-Diffusion-V1.4 in our paper.

Put your personalized collections in `path/to/personalized_collections`.

Train the model using the following command. 

    export OUTPUT_DIR="toy"
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file default_config.yaml train_dreambooth_dpg.py \
    --pretrained_model_name_or_path path/to/pretrained_stable_diffusion \
    --instance_data_dir path/to/personalized_collections \
    --instance_prompt "a photo of sks backpack" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --class_data_dir="path_class_images_backpack" \
    --output_dir=$OUTPUT_DIR \
    --class_prompt="a photo of backpack" \
    --resolution=512 --train_batch_size=1 --max_train_steps=1000 --learning_rate=1e-6  \
    --num_class_images=8 --lr_warmup_steps=0 \
    --lr_scheduler="constant" \
    --train_text_encoder
    
### Using 'DINO' reward

Download `ViT-S/16` ckpt from the official website `https://github.com/facebookresearch/dino`.


## **Inference**
Use the following command for inference

    CUDA_VISIBLE_DEVICES=0 python generate_images.py --ckpt_path /path/to/model --prompt "A sks backpack on the beach"

## **Visualization Examples**
<img width="852" alt="image" src="https://github.com/user-attachments/assets/a1ba1687-8864-4c19-872b-a1fef50c51f6">

## **Todo**
- [ ] Code of DINO reward of DreamBooth | Doing
- [ ] Code of face reward of DreamBooth
- [ ] Code of Look forward of CustomDiff
- [ ] Code of DINO reward of CustomDiff
<!-- ## **Citation**
    -->
