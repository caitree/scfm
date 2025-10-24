import itertools
import json
import os
import random
from collections import defaultdict

import torch
import tqdm
from datasets import Dataset, load_dataset
from datasets.features import Image, Value
from diffusers import FluxPipeline, StableDiffusion3Pipeline

from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

BUFFER_SIZE = 5000
PROMPTS_DATASETS = "Gustavosta/Stable-Diffusion-Prompts"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
SD3_5_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
IMAGE_EDIT_DATASETS = "iitolstykh/NHR-Edit"

ASPECT_RATIO_1024_BIN_SHORT = {
    "1.0": [1024.0, 1024.0],  # 1:1 # 16:16
    "1.33": [1280.0, 960.0],  # 4:3 # 20:15
}

ASPECT_RATIO_1024_BIN = {
    "0.57": [768.0, 1344.0],  # 4:7 # 12:21
    "0.66": [896.0, 1344.0],  # 2:3 # 14:21
    "0.75": [960.0, 1280.0],  # 3:4 # 15:20
    "1.0": [1024.0, 1024.0],  # 1:1 # 16:16
    "1.33": [1280.0, 960.0],  # 4:3 # 20:15
    "1.5": [1344.0, 896.0],  # 3:2 # 21:14
    "1.75": [1344.0, 768.0],  # 7:4 # 21:12
}

ASPECT_RATIO_512_BIN = {
    "0.6": [384.0, 640.0],  # 4:7 # 6: 10
    "0.78": [448.0, 576.0],  # 3:4 # 7:9
    "1.0": [512.0, 512.0],  # 1:1 # 8:8
    "1.29": [576.0, 448.0],  # 4:3 # 9:7
    "1.67": [640.0, 384.0],  # 3:2 # 10:6
}

def prompt_datasets(n_samples: int = 100):
    ds = load_dataset(PROMPTS_DATASETS, split="train", streaming=True)
    ds = ds.rename_column("Prompt", "prompt")
    return ds.shuffle(buffer_size=BUFFER_SIZE).take(n_samples)


@torch.no_grad()
def prepare_datasets(pipeline, n_samples: int = 100, **pipeline_kwargs):
    ds = prompt_datasets(n_samples=n_samples)

    images = []
    prompts = []
    for sample in tqdm.tqdm(ds, desc="Generating images"):
        image = pipeline(prompt=sample["prompt"], **pipeline_kwargs).images[0]
        images.append(image)
        prompts.append(sample["prompt"])
    new_ds = Dataset.from_dict({"image": images, "prompt": prompts})
    new_ds = new_ds.cast_column("image", feature=Image(decode=True))
    new_ds = new_ds.cast_column("prompt", feature=Value(dtype="string"))
    return new_ds

def export_image_caption(ds, data_dir: str, image_format: str = "webp"):
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "metadata.jsonl")):
        # remove metadata.jsonl
        os.remove(os.path.join(data_dir, "metadata.jsonl"))

    image_paths = []
    prompts = []
    for i, sample in tqdm.tqdm(enumerate(ds), desc="save images"):
        image, prompt = sample["image"], sample["prompt"]
        path = os.path.join(data_dir, f"{i}.{image_format}")
        image.save(path)
        image_paths.append(os.path.relpath(path, data_dir))
        prompts.append(prompt)
    new_ds = Dataset.from_dict({"file_name": image_paths, "prompt": prompts})
    new_ds = new_ds.cast_column("file_name", feature=Value(dtype="string"))
    new_ds = new_ds.cast_column("prompt", feature=Value(dtype="string"))
    new_ds.to_json(os.path.join(data_dir, "metadata.jsonl"))