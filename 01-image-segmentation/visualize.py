import torch
import segmentation_models_pytorch as smp
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128

# Transforms
image_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.PILToTensor(),
])

# Dataset
class PetSegmentation(OxfordIIITPet):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        image = image_transform(image)
        mask = mask_transform(mask).float()
        mask = (mask == 2).float()
        return image, mask

dataset = PetSegmentation(
    root="./data",
    split="test",
    target_types="segmentation",
    download=False
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model (change to resnet18 if needed)
backbone = "resnet34"

model = smp.Unet(
    encoder_name=backbone,
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(DEVICE)

model.load_state_dict(torch.load(f"model_{backbone}.pth", map_location=DEVICE))
model.eval()

images, masks = next(iter(loader))
images = images.to(DEVICE)

with torch.no_grad():
    outputs = model(images)

pred = torch.sigmoid(outputs[0]).cpu()
pred = (pred > 0.5).float()

# Plot
plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(images[0].permute(1,2,0).cpu())

plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(masks[0][0], cmap="gray")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred[0], cmap="gray")

plt.tight_layout()
plt.savefig(f"segmentation_{backbone}.png")
plt.close()

print(f"Saved segmentation_{backbone}.png")
