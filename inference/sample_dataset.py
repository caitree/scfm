import argparse
import os

from datasets import concatenate_datasets
from diffusers import FluxPipeline, StableDiffusion3Pipeline

from src.utils.sample_data import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_1024_BIN_SHORT,
    export_image_caption,
    prepare_datasets,
    prompt_datasets,
)

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
SD3_5_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


def parse_args():
    parser = argparse.ArgumentParser()
    # [flux, sd35]
    parser.add_argument(
        "--model",
        type=str,
        default="flux",
        choices=["flux", "sd35"],
        help="diffusion model to use",
    )
    parser.add_argument("--n_sample_per_bucket", type=int, default=8)
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 1024],
        help="resolution of the images",
    )
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    import torch
    import tqdm

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.model == "flux":
        pipeline = FluxPipeline.from_pretrained(
            FLUX_MODEL_ID, torch_dtype=dtype
        ).to(
            device=device,
        )
        print(f"Using {FLUX_MODEL_ID} model")
    elif args.model == "sd35":
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            SD3_5_MODEL_ID, torch_dtype=dtype
        ).to(
            device=device,
        )
        print(f"Using {SD3_5_MODEL_ID} model")
    else:
        raise ValueError(f"Invalid model: {args.model}")

    if args.resolution == 512:
        bucket = ASPECT_RATIO_512_BIN
    elif args.resolution == 1024:
        bucket = ASPECT_RATIO_1024_BIN_SHORT
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    n_sample_per_bucket = args.n_sample_per_bucket

    output_dir = os.path.join(args.output_dir)
    buckets = [[int(i[0]), int(i[1])] for i in bucket.values()]
    dataset_list = []
    n_sample = n_sample_per_bucket * len(buckets)
    pbar = tqdm.tqdm(total=n_sample, desc="Sampling")

    for bucket in buckets:
        pipeline_kwargs = {
            "num_inference_steps": 32,
            "guidance_scale": 3.5,
            "width": bucket[0],
            "height": bucket[1],
        }
        ds = prepare_datasets(
            pipeline, n_samples=n_sample_per_bucket, **pipeline_kwargs
        )
        dataset_list.append(ds)
        pbar.update(n_sample_per_bucket)
    ds = concatenate_datasets(dataset_list)
    export_image_caption(ds, data_dir=output_dir)
