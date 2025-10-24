import cv2
import torch
import numpy as np
import random
import os
import logging
import gc
import einops
import math
import sys

from math import ceil
from torchvision import transforms
from PIL import Image
from typing import (
    Tuple,
    Union,
    Dict,
)

from diffusers.loaders import FluxLoraLoaderMixin

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    import intel_extension_for_pytorch as ipex  # noqa

    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False

def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

logger = logging.getLogger(__name__)


def get_noise(size, seed=None, device="cpu", dtype=torch.bfloat16, layout=None):
    # generator = torch.manual_seed(seed)
    if seed == 0:
        seed = random.randint(0, 1000000000)
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    layout = layout or torch.strided
    return torch.randn(size, dtype=dtype, generator=generator, device=device, layout=layout)

def post_process_img(images):
    out_image = []
    for image in images:
        image = image.clamp(-1, 1)
        image = image.permute(1, 2, 0)
        image = (127.5 * (image + 1.0)).float().cpu().numpy().astype(np.uint8)
        out_image.append(Image.fromarray(image))
    return out_image

def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console
                from rich.logging import RichHandler

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)


def encode_image(imgpath, vae, default_max_size=768, specified_size=None) -> torch.Tensor:
    if isinstance(imgpath, str):
        cond_image = Image.open(imgpath).convert("RGB")
    elif isinstance(imgpath, Image.Image):
        cond_image = imgpath
    else:
        raise ValueError("imgpath must be a str or Image.Image")

    width, height = cond_image.size
    if specified_size is not None:
        infer_width, infer_height = specified_size
    else:
        infer_width, infer_height = largest_size_below_maxarea(width, height, default_max_size**2, 32) 
        infer_width = max(64, infer_width - infer_width % 32)  # round to divisible by 32
        infer_height = max(64, infer_height - infer_height % 32)  # round to divisible by 32
        
    if not (height == infer_height and width == infer_width):
        cond_image = cond_image.resize((infer_width, infer_height), Image.LANCZOS)
        # cond_image = resize_large_size_images(cond_image, (infer_width, infer_height))

    cond_image = IMAGE_TRANSFORMS(cond_image)
    if cond_image.ndim == 3:
        cond_image = cond_image.unsqueeze(0)

    with torch.no_grad():
        cond_image = cond_image.to(vae.device, vae.dtype)
        cond_latent = vae.encode(cond_image).latent_dist.sample()
        cond_latent = (cond_latent - vae.config.shift_factor) * vae.config.scaling_factor        
    return cond_latent

def lora_lora_state_dict(
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    return_alphas: bool = False,
    **kwargs,
):
    kwargs["return_lora_metadata"] = True
    state_dict, network_alphas, metadata = FluxLoraLoaderMixin.lora_state_dict(
        pretrained_model_name_or_path_or_dict, return_alphas=return_alphas, **kwargs
    )
    has_lora_keys = any("lora" in key for key in state_dict.keys())
    if not has_lora_keys:
        raise ValueError("Invalid LoRA checkpoint.")

    return state_dict, network_alphas, metadata


def dit_lora_merge(dit, lora_paths, names=None) -> None:
    adapter_weights = []
    adapter_names = []
    lora_paths = lora_paths.split(' ')
    for i, lora_path in enumerate(lora_paths):
        if ";" in lora_path:
            lora_path, lora_scale = lora_path.split(";")
            lora_scale = 1.0 if lora_scale=='' else float(lora_scale)
        else:
            lora_scale = 1.0

        adapter_weights.append(float(lora_scale))
        # adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
        adapter_name = names[i] if names is not None else str(i)
        adapter_names.append(adapter_name)
        lora_state_dict, network_alphas, metadata = lora_lora_state_dict(lora_path, return_alphas=True)

        dit.load_lora_adapter(
            lora_state_dict,
            network_alphas=network_alphas,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=None,
            low_cpu_mem_usage=False,
        )
    dit.set_adapters(adapter_names=adapter_names, weights=adapter_weights)

    # fuse LoRAs and unload weights
    # dit.fuse_lora(adapter_names=["0", "1"], lora_scale=1.0)
    # dit.delete_adapters(adapter_names='0')
    # dit.delete_adapters(adapter_names='1')

def dit_unload_lora_weights(transformer):
    transformer.unload_lora()
    if hasattr(transformer, "_transformer_norm_layers") and transformer._transformer_norm_layers:
        transformer.load_state_dict(transformer._transformer_norm_layers, strict=False)
        transformer._transformer_norm_layers = None


def prepare_img_ids(packed_latent_height: int, packed_latent_width: int, step: int = 1):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(0, packed_latent_height * step, step)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(0, packed_latent_width * step, step)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> (h w) c")
    return img_ids


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x

def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def round_to_steps(x, reso_steps):
    x = int(x + 0.5)
    return x - x % reso_steps

def largest_size_below_maxarea(current_width: int, current_height: int, target_area: int, reso_steps: int) -> Tuple[int, int]:
    aspect_ratio = current_width / current_height
    current_area = current_width * current_height
    
    # Calculate ideal dimensions based on target area
    ideal_width = math.sqrt(target_area * aspect_ratio)
    ideal_height = target_area / ideal_width
    assert abs(ideal_width / ideal_height - aspect_ratio) < 1e-2, "aspect ratio is illegal"

    # Try both width-first and height-first approaches
    # Width-first approach
    width_rounded = round_to_steps(ideal_width, reso_steps)
    height_from_width = round_to_steps(width_rounded / aspect_ratio, reso_steps)
    ar_width_rounded = width_rounded / height_from_width
    # area_width_approach = width_rounded * height_from_width

    # Height-first approach
    height_rounded = round_to_steps(ideal_height, reso_steps)
    width_from_height = round_to_steps(height_rounded * aspect_ratio, reso_steps)
    ar_height_rounded = width_from_height / height_rounded
    # area_height_approach = width_from_height * height_rounded

    # Choose the approach that better maintains aspect ratio
    if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
        final_width = width_rounded
        final_height = height_from_width
        final_area = final_width * final_height
    else:
        final_width = width_from_height
        final_height = height_rounded
        final_area = final_width * final_height

    return (final_width, final_height)


def resize_large_size_images(image: np.ndarray, target_size: Tuple[int, int], step_scale=0.6, interpolation=cv2.INTER_AREA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize large images using a stepwise reduction strategy.

    Parameters:
        image: Input image (NumPy array).
        target_size: Target size (width, height).
        interpolation: Interpolation method, default is cv2.INTER_AREA.
        step_scale: Scale factor for each reduction step, default is 0.6 (reduces by half each time).

    Returns:
        Resized image.
    """
    # Get original image dimensions
    image = np.array(image)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # If target size is larger than original size, directly use cv2.resize
    if target_w >= w or target_h >= h:
        return cv2.resize(image, target_size, interpolation=interpolation)
    # Stepwise image reduction
    while True:
        # Calculate current reduction ratio
        scale = max(target_w / w, target_h / h)

        # If current size is close to target size, directly resize to target size
        if scale >= step_scale:
            return cv2.resize(image, target_size, interpolation=interpolation)

        # Reduce image by step_scale
        new_w = int(w * step_scale)
        new_h = int(h * step_scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Update current dimensions
        h, w = new_h, new_w