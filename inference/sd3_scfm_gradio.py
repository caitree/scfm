import torch
import os
import argparse
import accelerate
import time
import logging
import numpy as np
import gradio as gr

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel
)
from src.utils.sd3_utils import encode_prompt
from src.utils.utils import (
    get_noise,
    post_process_img,
    clean_memory,
    dit_lora_merge,
    dit_unload_lora_weights,
)
from typing import List
from tqdm import tqdm

logger = logging.getLogger(__name__)

device = torch.device("cuda")
dtype = torch.bfloat16
accelerator = accelerate.Accelerator(mixed_precision="bf16")
# nega_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
nega_prompt = ""
# prefix="high resolution, photorealisitic, best quality, ultra-detailed, 4k. "
prefix=""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd3_type", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--port", type=int, default=8016)
    args = parser.parse_args()

    return args

args = parse_args()

clip_l_tokenizers = CLIPTokenizer.from_pretrained(
    args.sd3_type,
    subfolder="tokenizer",
)
clip_g_tokenizers = CLIPTokenizer.from_pretrained(
    args.sd3_type,
    subfolder="tokenizer_2",
)
t5xxl_tokenizers = T5TokenizerFast.from_pretrained(
    args.sd3_type,
    subfolder="tokenizer_3",
)
clip_l = CLIPTextModelWithProjection.from_pretrained(
    args.sd3_type,
    subfolder="text_encoder",
)
clip_g = CLIPTextModelWithProjection.from_pretrained(
    args.sd3_type,
    subfolder="text_encoder_2",
)
t5xxl = T5EncoderModel.from_pretrained(
    args.sd3_type,
    subfolder="text_encoder_3",
)
tokenizers = [clip_l_tokenizers, clip_g_tokenizers, t5xxl_tokenizers]
text_encoders = [clip_l, clip_g, t5xxl]
clip_l.to(device, dtype)
clip_g.to(device, dtype)
t5xxl.to(device, dtype)

vae = AutoencoderKL.from_pretrained(
    args.sd3_type,
    subfolder="vae",
)
vae.to(device)

mmdit = SD3Transformer2DModel.from_pretrained(
    args.sd3_type,
    subfolder="transformer",
)
mmdit.to(device, dtype)

batch_size = 1

def encode_prompts(prompts):
    with accelerator.autocast(), torch.no_grad():
        logger.info(f"Encoding prompts: {prompts}")
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length=256, device=device
        )
        prompt_embeds = prompt_embeds.to(device, dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype)

    prompt_embeds = torch.cat([prompt_embeds]*batch_size, dim=0)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds]*batch_size, dim=0)

    return [prompt_embeds, pooled_prompt_embeds]

logger.info(f'Encoding negative prompt: {nega_prompt}')
neg_prompt_embeds = encode_prompts(nega_prompt)


def denoise(
    prompt: List[torch.Tensor],
    width: int,
    height: int,
    cfg_scale: float,
    shift: float,
    steps: int,
    seed: int = 0
):
    do_cfg = (cfg_scale > 0)
    B, C, H, W = batch_size, 16, height//16*16//8, width//16*16//8
    latent = get_noise((B, C, H, W), seed, device, dtype)

    timesteps = torch.linspace(1.0, 0, steps+1).to(device=device, dtype=torch.float32)
    timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
    timesteps[0] = 1-1e-5

    logger.info(f"Inferece timesteps are {timesteps.cpu().numpy()}")
    t1=time.time()
    with accelerator.autocast(), torch.no_grad():
        for i, t in enumerate(tqdm(timesteps[:-1])):
            pred = mmdit(
                hidden_states=torch.cat([latent]*2) if do_cfg else latent,
                timestep=(t.expand(B*2) if do_cfg else t.expand(B))*1000,
                encoder_hidden_states=torch.cat([prompt[0], neg_prompt_embeds[0]]) if do_cfg else prompt[0],
                pooled_projections=torch.cat([prompt[1], neg_prompt_embeds[1]]) if do_cfg else prompt[1],
                return_dict=False,
            )[0]
            if do_cfg:
                pred, pred_nega = pred.chunk(2)
                pred = pred + cfg_scale * (pred - pred_nega)
            latent = latent + (timesteps[i+1] - timesteps[i]) * pred

    t2 = time.time()
    logger.info(f"Inferece time is {t2-t1}")
    denoised_latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    return denoised_latent.to(vae.dtype)

def main(lora_paths, prompt, width, height, cfg, shift, steps, seed):
    global global_lora_paths, global_prompt, global_prompt_embeds, neg_prompt_embeds

    if prompt != global_prompt:
        global_prompt = prompt
        global_prompt_embeds = encode_prompts(prefix+prompt)

    local_lora_paths=set()

    lora_paths = lora_paths.strip().split()
    for lora_path in lora_paths:
        if not 'safetensors' in lora_path:
            if ';' in lora_path:
                lora_path = lora_path.split(';')[0] + '/pytorch_lora_weights.safetensors;' + lora_path.split(';')[-1]
            else:
                lora_path = lora_path + '/pytorch_lora_weights.safetensors;1'
        local_lora_paths.add(lora_path)

    if local_lora_paths != global_lora_paths:
        global_lora_paths = local_lora_paths.copy()

        logger.info("Removing previous LoRAs weights...")
        dit_unload_lora_weights(mmdit)

        if len(local_lora_paths) > 0:
            logger.info(f"Merging LoRAs from {global_lora_paths}")
            dit_lora_merge(mmdit, ' '.join(list(global_lora_paths)))

    denoised_latent = denoise(global_prompt_embeds, int(width), int(height), cfg, shift, steps, seed)

    with torch.no_grad():
        img = vae.decode(denoised_latent, return_dict=False)[0]
        img = post_process_img(img)
    clean_memory()
    return img[0]

if __name__ == "__main__":
    global_lora_paths, global_prompt, global_prompt_embeds = set(), '', None

    with gr.Blocks() as server:
        with gr.Row():
            gr.Markdown("## scfm")
        with gr.Row():
            with gr.Column():
                lora_paths = gr.Textbox(label="LoRA Path", value="output/sd3-scfm/checkpoint-6000/pytorch_lora_weights.safetensors")
                prompt = gr.TextArea(label="Prompt", value="a cute squared planet with a blue colored surface surronded by a soft pink hued space background with glowing stars the whole scene has whimsical and adorable vibe with a bright shny and brilliant surface on the planet featuring singular charming house.")
                width = gr.Slider(minimum=128, maximum=2048, value=1024, step=64, label="width")
                height = gr.Slider(minimum=128, maximum=2048, value=1024, step=64, label="height")
                cfg = gr.Slider(minimum=0, maximum=8, value=4, step=0.1, label="CFG")
                shift = gr.Slider(minimum=1, maximum=10, value=3, step=0.1, label="Shift")
                steps = gr.Slider(minimum=1, maximum=50, value=8, step=1, label="Steps")
                seed = gr.Slider(minimum=0, maximum=100000000, value=0, step=1, label="Seed")

                interface_button = gr.Button("Generate Image")
            with gr.Column():
                gallery0 = gr.Image(type="pil", format="webp", label="output")
        interface_button.click(fn=main,
            inputs=[
                lora_paths,
                prompt,
                width,
                height,
                cfg,
                shift,
                steps,
                seed,
            ],
            outputs=[
                gallery0,
            ],
            api_name="scfm",
        )

    server.queue(max_size=100, status_update_rate=1)
    server.launch(server_name="0.0.0.0", server_port=args.port, max_threads=1)