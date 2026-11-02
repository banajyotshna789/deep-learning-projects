import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cond_model import ConditionalUNet, DDPM
from utils import save_image_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

net = ConditionalUNet(base_channels=96, time_dim=256).to(device)
ddpm = DDPM(net, timesteps=1000, device=device).to(device)

opt = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

epochs = 30

for epoch in range(1, epochs + 1):
    ddpm.train()
    running = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device).long()

        opt.zero_grad()
        loss = ddpm.loss(x, y)
        loss.backward()
        opt.step()

        running += loss.item()

        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    avg_loss = running / len(loader)
    print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    ckpt_path = f"outputs/cond_epoch_{epoch}.pt"
    torch.save(ddpm.state_dict(), ckpt_path)

    sample_labels = torch.arange(10, device=device)
    samples = ddpm.sample((10, 1, 28, 28), sample_labels)
    save_image_grid(samples, f"outputs/cond_samples_epoch_{epoch}.png", nrow=10)

    print(f"Saved {ckpt_path}")

print("Training complete.")
