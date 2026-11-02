import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from gan.generator import Generator
from gan.discriminator import Discriminator

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 64
z_dim = 100
lr = 1e-4
epochs = 50
lambda_gp = 10
n_critic = 5
num_classes = 10

sample_dir = "generated_samples"
ckpt_dir = "checkpoints"
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# -----------------------------
# Models
# -----------------------------
G = Generator().to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

# Fixed noise for visualization
fixed_noise = torch.randn(num_classes, z_dim, device=device)
fixed_labels = torch.arange(0, num_classes, device=device).long()

# -----------------------------
# Gradient Penalty
# -----------------------------
def gradient_penalty(D, real, fake, labels):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, labels)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# -----------------------------
# Save Samples
# -----------------------------
@torch.no_grad()
def save_samples(epoch):
    G.eval()
    fake = G(fixed_noise, fixed_labels)
    fake = (fake + 1) / 2
    utils.save_image(fake, f"{sample_dir}/epoch_{epoch}.png", nrow=5)
    G.train()

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    for i, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device).long()

        # ---------------------
        # Train Critic
        # ---------------------
        for _ in range(n_critic):
            z = torch.randn(real.size(0), z_dim, device=device)
            fake = G(z, labels)

            D_real = D(real, labels).mean()
            D_fake = D(fake.detach(), labels).mean()

            gp = gradient_penalty(D, real, fake.detach(), labels)

            loss_D = -(D_real - D_fake) + lambda_gp * gp

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = G(z, labels)

        loss_G = -D(fake, labels).mean()

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Step [{i}/{len(loader)}] "
                f"D Loss: {loss_D.item():.4f} "
                f"G Loss: {loss_G.item():.4f} "
                f"D(real): {D_real.item():.4f} "
                f"D(fake): {D_fake.item():.4f} "
                f"GP: {gp.item():.4f}"
            )

    save_samples(epoch + 1)

    torch.save(G.state_dict(), f"{ckpt_dir}/G_epoch_{epoch+1}.pth")
    torch.save(D.state_dict(), f"{ckpt_dir}/D_epoch_{epoch+1}.pth")

print("Training finished.")
