import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, embed_dim=32, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        x = self.project(x)
        x = x.view(x.size(0), 128, 7, 7)
        return self.net(x)


# -----------------------------
# Projection Critic
# -----------------------------
class Critic(nn.Module):
    def __init__(self, num_classes=10, img_channels=1, embed_dim=128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, embed_dim)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, img, labels):
        h = self.features(img)
        h = self.flatten(h)
        h = self.fc(h)  # [B, embed_dim]

        out = self.out(h).squeeze(1)

        y = self.label_emb(labels)  # [B, embed_dim]
        proj = torch.sum(h * y, dim=1)

        return out + proj


# -----------------------------
# Utils
# -----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic, real, fake, labels, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    scores = critic(interpolated, labels)
    grad_outputs = torch.ones_like(scores, device=device)

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


@torch.no_grad()
def save_samples(G, device, fixed_z, fixed_labels, out_path, epoch=None):
    G.eval()
    fake = G(fixed_z.to(device), fixed_labels.to(device))
    grid = utils.make_grid(fake, nrow=10, normalize=True, value_range=(-1, 1))
    out_path.mkdir(parents=True, exist_ok=True)
    if epoch is None:
        file_path = out_path / "initial.png"
    else:
        file_path = out_path / f"samples_epoch_{epoch:03d}.png"
    utils.save_image(grid, file_path)
    G.train()


# -----------------------------
# Train
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    G = Generator(latent_dim=args.latent_dim).to(device)
    C = Critic().to(device)

    G.apply(weights_init)
    C.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(100, args.latent_dim, device=device)
    fixed_labels = torch.tensor([i for i in range(10) for _ in range(10)], dtype=torch.long, device=device)

    save_samples(G, device, fixed_z, fixed_labels, sample_dir)

    for epoch in range(1, args.epochs + 1):
        G.train()
        C.train()

        c_loss_sum = 0.0
        g_loss_sum = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for real_imgs, real_labels in pbar:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_imgs.size(0)

            # -------------------------
            # Train Critic more times
            # -------------------------
            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                fake_imgs = G(z, fake_labels).detach()

                real_scores = C(real_imgs, real_labels)
                fake_scores = C(fake_imgs, fake_labels)

                gp = gradient_penalty(C, real_imgs, fake_imgs, real_labels, device)

                c_loss = fake_scores.mean() - real_scores.mean() + args.gp_lambda * gp

                opt_C.zero_grad(set_to_none=True)
                c_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(C.parameters(), args.grad_clip)
                opt_C.step()

            # -------------------------
            # Train Generator
            # -------------------------
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (batch_size,), device=device)
            gen_imgs = G(z, gen_labels)
            g_loss = -C(gen_imgs, gen_labels).mean()

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_G.step()

            c_loss_sum += c_loss.item()
            g_loss_sum += g_loss.item()

            pbar.set_postfix({
                "C_loss": f"{c_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}"
            })

        avg_c = c_loss_sum / len(loader)
        avg_g = g_loss_sum / len(loader)
        print(f"Epoch [{epoch}/{args.epochs}]  C_loss: {avg_c:.4f}  G_loss: {avg_g:.4f}")

        save_samples(G, device, fixed_z, fixed_labels, sample_dir, epoch=epoch)

        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "critic_state_dict": C.state_dict(),
            "opt_G_state_dict": opt_G.state_dict(),
            "opt_C_state_dict": opt_C.state_dict(),
            "args": vars(args),
        }, ckpt_dir / f"cwagp_epoch_{epoch:03d}.pt")

    torch.save(G.state_dict(), out_dir / "generator_final.pt")
    torch.save(C.state_dict(), out_dir / "critic_final.pt")
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional WGAN-GP on MNIST")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs_cwagp")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
