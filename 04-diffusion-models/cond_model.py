import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999)

def extract(a, t, x_shape):
    out = a.index_select(0, t)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, cond):
        x = self.res(x, cond)
        skip = x
        down = self.down(x)
        return skip, down

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res = ResBlock(out_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, cond)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=96, time_dim=256, num_classes=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.label_emb = nn.Embedding(num_classes, time_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, time_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim)

        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)

        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up2 = UpBlock(base_channels, base_channels, base_channels, time_dim)

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 1)
        )

    def forward(self, x, t, y):
        cond = self.time_mlp(t) + self.label_emb(y)
        x = self.init_conv(x)

        skip1, x = self.down1(x, cond)
        skip2, x = self.down2(x, cond)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        x = self.up1(x, skip2, cond)
        x = self.up2(x, skip1, cond)

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

    def loss(self, x0, y):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t, y)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, shape, y):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t, y)

            beta_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

            model_mean = sqrt_recip_alphas_t * (x - beta_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

            if i > 0:
                posterior_var_t = extract(self.posterior_variance, t, x.shape)
                x = model_mean + torch.sqrt(posterior_var_t) * torch.randn_like(x)
            else:
                x = model_mean

        return x
