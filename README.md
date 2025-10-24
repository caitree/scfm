# SCFM: Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch

## Overview
We introduce SCFM — a highly efficient post-training distillation method that converts many pre-trained flow matching diffusion model (e.g., Flux, SD3, etc) into a 3–8 step sampler in <1 A100 day.

## Prerequisites
```bash

pip install torch==2.6 torchvision==0.21 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Data Prepare

### prepare flux data
generate custom datasets by diffusers
```bash 
export HF_TOKEN="your_huggingface_token"
python inference/sample_dataset.py \
--output_dir flux_dataset_1024 \
--resolution 1024 \
--num_inference_steps 32 \
--n_sample_per_bucket 8

```
   
## Training
```bash
bash scripts/train_flux_scfm.sh
```

## Inference
After training, you can perform highly-customizable inference with gradio, e.g., kicking around with lora scales, timesteps shifting, cfg, or even merging with off-the-shelf loras, and etc.
  ```bash
  python inference/flux_scfm_gradio.py --port xxxx
  ```