import torch
import numpy as np
import math
import random

from typing import Tuple, Callable
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_scfm_timesteps_flux(args, t_skip, latents, noise, device, dtype):
    bsz, _, h, w = latents.shape
    k = math.ceil(bsz * args.non_shortcut_ratio)

    teacher_sigmas = torch.ones((bsz,), dtype=torch.float32)
    teacher_dt = torch.ones((bsz, t_skip), dtype=torch.float32)
    student_dt = torch.ones((bsz,), dtype=torch.float32)
    teacher_indices = []
    self_indices = []

    teacher_min_timesteps = max(args.teacher_min_timesteps, t_skip)
    for i in range(bsz):
        if np.random.rand() < args.teacher_ratio: # teacher
            tts = random.choice(torch.arange(teacher_min_timesteps, args.teacher_max_timesteps + t_skip, t_skip))
            teacher_indices.append(i)
        else: # self
            power = int(torch.log2(torch.tensor(args.teacher_min_timesteps)).item())
            tts = random.choice(1 << torch.arange(1, power))

            self_indices.append(i)

        timesteps = torch.linspace(1.0, 0, tts+1)
        if np.random.rand() < 0.1: # perform dynamic shift
            mu = get_lin_function(y1=0.5, y2=1.15)((w // 2) * (h // 2))
            timesteps = time_shift(mu, 1.0, timesteps)
        else: # perform fixed shift
            shift = np.random.rand() * 2 + 2.5
            timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

        timesteps[0] = 1 - 1e-5 # avoid being pure noise
        start_idx = random.choice(torch.arange(0, tts, t_skip))

        teacher_sigmas[i] = timesteps[start_idx]
        for j in range(t_skip):
            teacher_dt[i][j] = timesteps[start_idx+j] - timesteps[start_idx+j+1]
        student_dt[i] = timesteps[start_idx] - timesteps[start_idx+t_skip]

    if k > 0 : student_dt[:k] = teacher_dt[:k][0]

    teacher_sigmas = teacher_sigmas.to(device)
    teacher_dt = teacher_dt.to(device)
    student_dt = student_dt.to(device)

    noisy_model_input = (1 - teacher_sigmas.view(-1, 1, 1, 1)) * latents + teacher_sigmas.view(-1, 1, 1, 1) * noise
    target = (noise - latents).to(torch.float32)
    
    return noisy_model_input.to(dtype=dtype), target, teacher_sigmas, teacher_dt, student_dt, teacher_indices, self_indices