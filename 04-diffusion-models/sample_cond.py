import torch
from cond_model import ConditionalUNet, DDPM
from utils import save_image_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ConditionalUNet(base_channels=96, time_dim=256).to(device)
ddpm = DDPM(net, timesteps=1000, device=device).to(device)

ckpt = "outputs/cond_epoch_30.pt"
ddpm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
ddpm.eval()

# 🔥 Create 64 samples (8x8 grid)
num_images = 64

# Repeat labels 0–9 to fill 64 slots
labels = torch.arange(10).repeat((num_images // 10) + 1)[:num_images]
labels = labels.to(device)

with torch.no_grad():
    samples = ddpm.sample((num_images, 1, 28, 28), labels)

# Save as 8x8 grid
save_image_grid(samples, "outputs/final_cond_8x8.png", nrow=8)

print("saved outputs/final_cond_8x8.png")
