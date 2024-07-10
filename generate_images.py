# generate dreambooth benchmark images via its corresponding prompts
import argparse, os
from tqdm import tqdm, trange
import time

from torchvision.utils import make_grid
from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import CLIPTextModel

# for example
prompt = "A photo of sks backpack"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")
    
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A photo of sks backpack", 
        help="prompt to generate images")

    unique_token = "sks"
    
    opt = parser.parse_args()
    
    # If passed along, set the training seed now.
    if opt.seed is not None:
        set_seed(opt.seed)

    ckpt_path = opt.ckpt_path
        
    model_id = ckpt_path
    
    # can specify the ckpt
    text_encoder_path = os.path.join(ckpt_path, "checkpoint-1000", "text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)

    unet_path = os.path.join(ckpt_path, "checkpoint-1000", "unet")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)

    generator = torch.Generator("cuda").manual_seed(23)
    
    pipe = DiffusionPipeline.from_pretrained(model_id, text_encoder=text_encoder, unet=unet, torch_dtype=torch.float16).to("cuda")
    
    # the following two line just for example
    prompt_list = []
    prompt_list.append(opt.prompt)
    
    ## you can un-comment to evaluate dreambooth benchmark
    # dataset_name = ckpt_path.split("dbp_")[1]
    # dataset_name = dataset_name.replace("/", "")
    # if dataset_name in ["cat", "dog"]:
    #     is_live = True 
    # else:
    #     is_live = False
    
    # # here to be refined
    # class_token = dataset_name
    # subject_name_class_lists = {
    #     "wolf_plushie": "stuffed animal",
    #     "backpack": "backpack",
    #     "backpack_dog": "backpack",
    #     "bear_plushie": "stuffed animal",
    #     "berry_bowl": "bowl",
    #     "can": "can",
    #     "candle": "candle",
    #     "cat": "cat",
    #     "cat2": "cat",
    #     "clock": "clock",
    #     "colorful_sneaker": "sneaker",
    #     "dog": "dog",
    #     "dog2": "dog",
    #     "dog3": "dog",
    #     "dog5": "dog",
    #     "dog6": "dog",
    #     "dog7": "dog",
    #     "dog8": "dog",
    #     "duck_toy": "toy",
    #     "fancy_boot": "boot",
    #     "grey_sloth_plushie": "stuffed animal",
    #     "monster_toy": "toy",
    #     "pink_sunglasses": "glasses",
    #     "poop_emoji": "toy",
    #     "rc_car": "toy",
    #     "red_cartoon": "cartoon",
    #     "robot_toy": "toy",
    #     "shiny_sneaker": "sneaker",
    #     "teapot": "teapot",
    #     "vase": "vase"
    # }
    # class_token = subject_name_class_lists[dataset_name]

    # if is_live:
    #     prompt_list = [
    #         'a {0} {1} in the jungle'.format(unique_token, class_token),
    #         'a {0} {1} in the snow'.format(unique_token, class_token),
    #         'a {0} {1} on the beach'.format(unique_token, class_token),
    #         'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
    #         'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
    #         'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
    #         'a {0} {1} with a city in the background'.format(unique_token, class_token),
    #         'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
    #         'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
    #         'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
    #         'a {0} {1} wearing a red hat'.format(unique_token, class_token),
    #         'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
    #         'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
    #         'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
    #         'a {0} {1} in a chef outfit'.format(unique_token, class_token),
    #         'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
    #         'a {0} {1} in a police outfit'.format(unique_token, class_token),
    #         'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
    #         'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
    #         'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
    #         'a red {0} {1}'.format(unique_token, class_token),
    #         'a purple {0} {1}'.format(unique_token, class_token),
    #         'a shiny {0} {1}'.format(unique_token, class_token),
    #         'a wet {0} {1}'.format(unique_token, class_token),
    #         'a cube shaped {0} {1}'.format(unique_token, class_token)
    #         ]
    # else:
    #     prompt_list = [
    #         'a {0} {1} in the jungle'.format(unique_token, class_token),
    #         'a {0} {1} in the snow'.format(unique_token, class_token),
    #         'a {0} {1} on the beach'.format(unique_token, class_token),
    #         'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
    #         'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
    #         'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
    #         'a {0} {1} with a city in the background'.format(unique_token, class_token),
    #         'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
    #         'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
    #         'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
    #         'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
    #         'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
    #         'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
    #         'a {0} {1} floating on top of water'.format(unique_token, class_token),
    #         'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
    #         'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
    #         'a {0} {1} on top of a mirror'.format(unique_token, class_token),
    #         'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
    #         'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
    #         'a {0} {1} on top of a white rug'.format(unique_token, class_token),
    #         'a red {0} {1}'.format(unique_token, class_token),
    #         'a purple {0} {1}'.format(unique_token, class_token),
    #         'a shiny {0} {1}'.format(unique_token, class_token),
    #         'a wet {0} {1}'.format(unique_token, class_token),
    #         'a cube shaped {0} {1}'.format(unique_token, class_token)
    #     ]
    
    start_time = time.time()
    save_image_folder = os.path.join(ckpt_path, "generate_imgs")
    for prompt_item in prompt_list:
        prompt = prompt_item
        
        prompt_path = os.path.join(save_image_folder, prompt.replace(" ", "-"))
        os.makedirs(prompt_path, exist_ok=True)
        print("Generate image: ", prompt, "-"*10)
        for i in range(8):
            image_save_path = os.path.join(prompt_path, str(i) + ".png")
            image = pipe(prompt, num_inference_steps=200, guidance_scale=7.5, generator=generator).images[0]
            image.save(image_save_path)
