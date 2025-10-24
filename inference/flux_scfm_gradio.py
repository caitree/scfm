import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import einops
import json
import random
import copy
import math
import argparse
import accelerate
import diffusers
import time
import logging
import gradio as gr
import numpy as np

from typing import List
from tqdm import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import (
    AutoencoderKL,
    FluxTransformer2DModel,
)
from src.utils.flux_utils import encode_prompt
from src.utils.utils import (
    encode_image,
    get_noise,
    setup_logging,
    post_process_img,
    clean_memory,
    dit_lora_merge,
    dit_unload_lora_weights,
    pack_latents,
    unpack_latents,
    prepare_img_ids,
    calculate_shift,
)
from src.mods.flux_teacache import teacache_forward
from src.utils.train_utils import (
    get_lin_function,
    time_shift,
)

logger = logging.getLogger(__name__)

device = torch.device("cuda")
dtype = torch.bfloat16

FluxTransformer2DModel.forward = teacache_forward

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux_type", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    return args

args = parse_args()

clip_l_tokenizers = CLIPTokenizer.from_pretrained(
    args.flux_type,
    subfolder="tokenizer",
)
t5xxl_tokenizers = T5TokenizerFast.from_pretrained(
    args.flux_type,
    subfolder="tokenizer_2",
)
clip_l = CLIPTextModel.from_pretrained(
    args.flux_type,
    subfolder="text_encoder",
)
t5xxl = T5EncoderModel.from_pretrained(
    args.flux_type,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
)
tokenizers = [clip_l_tokenizers, t5xxl_tokenizers]
text_encoders = [clip_l, t5xxl]
clip_l.to(device, dtype)
t5xxl.to(device)

vae = AutoencoderKL.from_pretrained(
    args.flux_type,
    subfolder="vae",
)
vae.to(device)

flux = FluxTransformer2DModel.from_pretrained(
    args.flux_type,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

accelerator = accelerate.Accelerator(mixed_precision="bf16")

flux.to(device)

batch_size = 1

def encode_prompts(prompts):
    with accelerator.autocast(), torch.no_grad():
        logger.info(f"Encoding prompts: {prompts}")
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length=256, device=device
        )
        prompt_embeds = prompt_embeds.to(device, dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype)
        text_ids = text_ids.to(device, dtype)

    prompt_embeds = torch.cat([prompt_embeds]*batch_size, dim=0)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds]*batch_size, dim=0)

    return [prompt_embeds, pooled_prompt_embeds], text_ids

nega_prompt = ""
logger.info(f'Encoding negative prompt: {nega_prompt}')
neg_prompt_embeds, _ = encode_prompts(nega_prompt)

def denoise(
    prompt: List[torch.Tensor],
    text_ids: torch.Tensor,
    width: int,
    height: int,
    cfg: float,
    true_cfg: float,
    shift: float,
    steps: int,
    seed: int = 0
):
    do_cfg = (true_cfg >0)
    B, C, H, W = batch_size, 16, height//8, width//8
    packed_H, packed_W, packed_C = H//2, W//2, C*4

    guidance = torch.tensor([cfg], device=device).expand(B*2) if do_cfg else torch.tensor([cfg], device=device).expand(B)
    
    latent = get_noise((B, packed_H*packed_W, packed_C), seed, device, torch.float32)
    img_ids = prepare_img_ids(packed_H, packed_W).to(latent)

    timesteps = torch.linspace(1.0, 0, steps+1).to(device=device, dtype=torch.float32)
    if shift==1:
        mu = get_lin_function()(packed_H*packed_W)
        timesteps = time_shift(mu, 1.0, timesteps)
    else:
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
    timesteps[0] = 1 - 1e-5

    logger.info(f"Inferece timesteps are {timesteps.cpu().numpy()}")
    t1 = time.time()
    with accelerator.autocast(), torch.no_grad():
        for i, t in enumerate(tqdm(timesteps[:-1])):
            pred = flux(
                hidden_states=torch.cat([latent]*2) if do_cfg else latent,
                timestep=(t.expand(B*2) if do_cfg else t.expand(B)),
                guidance=guidance if 'dev' in args.flux_type else None,
                encoder_hidden_states=torch.cat([prompt[0], neg_prompt_embeds[0]]) if do_cfg else prompt[0],
                pooled_projections=torch.cat([prompt[1], neg_prompt_embeds[1]]) if do_cfg else prompt[1],
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            if do_cfg:
                pred, pred_nega = pred.chunk(2)
                pred = pred + true_cfg * (pred - pred_nega)
            latent = latent + (timesteps[i+1] - timesteps[i]) * pred

    t2 = time.time()
    logger.info(f"Inferece time is {t2-t1}")

    denoised_latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    denoised_latent = unpack_latents(denoised_latent, packed_H, packed_W)
    return denoised_latent.to(vae.dtype)

def main(lora_paths, prompt, width, height, cfg, true_cfg, shift, steps, seed, thresh):
    global global_lora_paths, global_prompt, global_prompt_embeds, global_text_ids, scheduler, delta_t_embedder

    if prompt != global_prompt:
        global_prompt = prompt
        global_prompt_embeds, global_text_ids = encode_prompts(prompt)

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
        dit_unload_lora_weights(flux)
        # import pdb
        # pdb.set_trace()
        if len(local_lora_paths) > 0:
            logger.info(f"Merging LoRAs from {global_lora_paths}")
            dit_lora_merge(flux, ' '.join(list(global_lora_paths)))

    flux.__class__.cnt = 0
    flux.__class__.num_steps = steps
    flux.__class__.rel_l1_thresh = thresh # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    flux.__class__.accumulated_rel_l1_distance = 0
    flux.__class__.previous_modulated_input = None
    flux.__class__.previous_residual = None
    flux.__class__.enable_teacache = False if thresh == 0 else True

    denoised_latent = denoise(global_prompt_embeds, global_text_ids, int(width), int(height), cfg, true_cfg, shift, steps, seed)

    with torch.no_grad():
        img = vae.decode(denoised_latent, return_dict=False)[0]
        img = post_process_img(img)
    clean_memory()
    return img[0]

if __name__ == "__main__":
    global_lora_paths, global_prompt, global_prompt_embeds, global_text_ids = set(), '', None, None

    with gr.Blocks() as server:
        with gr.Row():
            gr.Markdown("## T2I Interface")
        with gr.Row():
            with gr.Column():
                lora_paths = gr.Textbox(label="LoRA Path", value="output/flux-scfm/checkpoint-10000/pytorch_lora_weights.safetensors;1.5")
                prompt = gr.TextArea(label="Prompt", value="An ultrarealistic, high-quality stock photo presents a majestic aerial view of a floating island kingdom, suspended amidst swirling clouds and bathed in ethereal light. Intricate bridges connect ornate towers and cascading gardens, while winged creatures soar gracefully through the celestial skies. The architecture speaks of advanced civilization and refined aesthetics, instilling a sense of wonder and igniting the imagination with endless narratives of heroic quests and untold legends.")
                width = gr.Slider(minimum=128, maximum=2048, value=1024, step=64, label="width")
                height = gr.Slider(minimum=128, maximum=2048, value=768, step=64, label="height")
                cfg = gr.Slider(minimum=0, maximum=15, value=5, step=0.1, label="CFG")
                true_cfg = gr.Slider(minimum=0, maximum=8, value=0, step=0.1, label="True CFG")
                shift = gr.Slider(minimum=1, maximum=5, value=3.25, step=0.1, label="shift")
                steps = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Steps")
                seed = gr.Slider(minimum=0, maximum=1000000, value=0, step=1, label="Seed")
                thresh = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label="teacache threshold")

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
                true_cfg,
                shift,
                steps,
                seed,
                thresh,
            ],
            outputs=[
                gallery0,
            ],
            api_name="scfm",
        )

    server.queue(max_size=100, status_update_rate=1)
    server.launch(server_name="0.0.0.0", server_port=args.port, max_threads=1)