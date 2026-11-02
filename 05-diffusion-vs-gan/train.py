import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import SimpleUNet, DDPM
from utils import save_image_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

net = SimpleUNet().to(device)
ddpm = DDPM(net, timesteps=1000, device=device).to(device)

opt = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

epochs = 30
save_every = 1

for epoch in range(1, epochs + 1):
    ddpm.train()
    running = 0.0

    for step, (x, _) in enumerate(loader):
        x = x.to(device)
        opt.zero_grad()
        loss = ddpm.loss(x)
        loss.backward()
        opt.step()
        running += loss.item()

        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    avg_loss = running / len(loader)
    print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    if epoch % save_every == 0:
        ckpt_path = f"outputs/ddpm_epoch_{epoch}.pt"
        torch.save(ddpm.state_dict(), ckpt_path)
        samples = ddpm.sample((16, 1, 28, 28))
        save_image_grid(samples, f"outputs/sample_epoch_{epoch}.png", nrow=4)
        print(f"Saved {ckpt_path}")

print("Training complete.")
