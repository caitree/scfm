# Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch [![arXiv](https://img.shields.io/badge/arXiv-2305.12345-b31b1b.svg)](https://arxiv.org/pdf/2510.17858)

## Overview
We introduce SCFM — a highly efficient post-training distillation method that converts many pre-trained flow matching diffusion model (e.g., Flux, SD3, etc) into a 3–8 step sampler in <1 A100 day.

## Prerequisites
```bash
pip install torch==2.6 torchvision==0.21 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Data Prepare

### prepare data
generate custom datasets by diffusers
```bash 
export HF_TOKEN="your_huggingface_token"

# for flux
python inference/sample_dataset.py \
--output_dir flux_dataset_1024 \
--resolution 1024 \
--num_inference_steps 32 \
--n_sample_per_bucket 8

# for sd35-large
python inference/sample_dataset.py \
--model sd35 \
--output_dir sd3_dataset_1024 \
--resolution 1024 \
--num_inference_steps 32 \
--n_sample_per_bucket 8
```
   
## Training
```bash
bash scripts/train_flux_scfm.sh

bash scripts/train_sd3_scfm.sh
```

For both models, the loss should remain around 0.0x~0.1x to indicate stable training.

## Inference
After training, you can perform highly-customizable inference with gradio, e.g., kicking around with lora scales, timesteps shifting, cfg, or even merging with off-the-shelf loras, and etc.

Or, simply download the checkpoints from[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow?)](https://huggingface.co/cxxai/scfm) or [![Civitai](https://img.shields.io/badge/Civitai-Model-blue?logo=civitai&logoColor=white)](https://civitai.com/models/2064593/shortcutfm-843-steps-loras), and run inference by providing the correct model path to the gradio script.
```bash
python inference/flux_scfm_gradio.py --port xxxx
```

Tips: Using fewer steps generally requires a higher LoRA scale, typically within the safe range of [1, 1.75]. A lower timestep shift tends to produce sharper details but may introduce artifacts, while a higher timestep shift can preserve overall structure better at the cost of potential blurriness.

The SD35 checkpoint is only trained with null (empty) negative prompts, so it is recommended to use null negative prompts when using it.

## What can you do with this? / What can be further optimized?
- Incorporating this into your pretrained models allows them to maintain or improve their output fidelity while simultaneously boosting generation speed.
- Fine-tune SCFM on your own dataset to accelerate generation specifically for the data domains you care about.
- Train without text encoders or VAEs by caching latents.
- Improve upon our checkpoints: we trained under very limited computational resources and used almost no training tricks, so you might be able to obtain even better results with larger batches, longer training, or higher resolution.
- Note on model parallelism: it is currently unsupported due to the LoRA–EMA implementation design, but contributions to enable this are very welcome.
- Resolution limits: the released checkpoints were trained at a maximum resolution of 512×512; higher-resolution training (e.g., >512) may yield further improvements.
- Model and modality flexibility: you can replace FLUX with other flow-matching models or even extend SCFM to different modalities such as 3D, video, or audio, since our implementation works seamlessly with the [![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97%20Diffusers-Repository-blue)](https://github.com/huggingface/diffusers) library.

## 📖 Citation
If you find our work useful, please consider citing our paper:
```bibtex
@inproceedings{scfm,
  title = {Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch},
  author = {Xu Cai and Yang Wu and Qianli Chen and Haoran Wu and Lichuan Xiang and Hongkai Wen},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025}
}
```