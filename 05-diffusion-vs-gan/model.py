import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import extract, cosine_beta_schedule

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        time_emb = self.mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block1 = Block(in_ch, out_ch, time_emb_dim)
        self.block2 = Block(out_ch, out_ch, time_emb_dim)
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        down = self.downsample(x)
        return x, down

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block1 = Block(out_ch + skip_ch, out_ch, time_emb_dim)
        self.block2 = Block(out_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        return x

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = Down(base_channels, base_channels, time_dim)
        self.down2 = Down(base_channels, base_channels * 2, time_dim)

        self.mid1 = Block(base_channels * 2, base_channels * 2, time_dim)
        self.mid2 = Block(base_channels * 2, base_channels * 2, time_dim)

        self.up1 = Up(base_channels * 2, base_channels, base_channels * 2, time_dim)
        self.up2 = Up(base_channels, base_channels, base_channels, time_dim)

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)

        skip1, x = self.down1(x, t)
        skip2, x = self.down2(x, t)

        x = self.mid1(x, t)
        x = self.mid2(x, t)

        x = self.up1(x, skip2, t)
        x = self.up2(x, skip1, t)

        return self.final(x)

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, device="cuda"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t)
            beta_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)

            model_mean = sqrt_recip_alpha_t * (x - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t)

            if i > 0:
                posterior_var_t = extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x = model_mean
        return x
