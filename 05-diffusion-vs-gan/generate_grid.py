import torch
from torchvision.utils import save_image
from gan.generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
num_samples = 64
num_classes = 10

# 👉 use BEST checkpoint (change if needed)
G = Generator().to(device)
G.load_state_dict(torch.load("checkpoints/G_epoch_35.pth", map_location=device))
G.eval()

z = torch.randn(num_samples, z_dim, device=device)

labels = torch.arange(0, num_classes, device=device).repeat(7)[:num_samples]
labels = labels.long()

with torch.no_grad():
    samples = G(z, labels)
    samples = (samples + 1) / 2

save_image(samples, "gan_8x8_grid.png", nrow=8)

print("Saved gan_8x8_grid.png")
