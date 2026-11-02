import torch
from model import SimpleUNet, DDPM
from utils import save_image_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = SimpleUNet().to(device)
ddpm = DDPM(net, timesteps=1000, device=device).to(device)

ckpt = "outputs/ddpm_epoch_10.pt"
ddpm.load_state_dict(torch.load(ckpt, map_location=device,weights_only=True))
ddpm.eval()

with torch.no_grad():
    samples = ddpm.sample((16, 1, 28, 28))

save_image_grid(samples, "outputs/final_samples.png", nrow=4)
print("saved outputs/final_samples.png")
