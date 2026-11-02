import math
import torch

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(0, t)   # keep t on the same device as a
    return out.view(b, *((1,) * (len(x_shape) - 1)))

def save_image_grid(x, path, nrow=8):
    import torchvision
    x = (x.clamp(-1, 1) + 1) / 2
    torchvision.utils.save_image(x, path, nrow=nrow)
