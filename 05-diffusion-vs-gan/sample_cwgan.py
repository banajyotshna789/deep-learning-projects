import torch
import matplotlib.pyplot as plt
from gan.generator import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
G.load_state_dict(torch.load("cwgan_generator.pt", map_location=device))
G.eval()

z = torch.randn(64, 100).to(device)
labels = torch.arange(0, 10).repeat(7)[:64].to(device)

with torch.no_grad():
    imgs = G(z, labels).cpu()

imgs = (imgs + 1) / 2

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i][0], cmap="gray")
    ax.axis("off")

plt.savefig("cwgan_samples.png")
