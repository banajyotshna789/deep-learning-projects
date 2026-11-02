import torch
import segmentation_models_pytorch as smp
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128

# -----------------------
# Transforms
# -----------------------
image_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.PILToTensor(),
])

# -----------------------
# Dataset
# -----------------------
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

# 🔹 Use SAME image for both models
sample_index = 10
image, mask = dataset[sample_index]
image_batch = image.unsqueeze(0).to(DEVICE)

# -----------------------
# Function to get prediction
# -----------------------
def get_prediction(backbone):

    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(f"model_{backbone}.pth", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        output = model(image_batch)

    pred = torch.sigmoid(output[0]).cpu()
    pred = (pred > 0.5).float()

    return pred

# -----------------------
# Get predictions
# -----------------------
pred18 = get_prediction("resnet18")
pred34 = get_prediction("resnet34")

# -----------------------
# Plot side-by-side comparison
# -----------------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Input")
plt.imshow(image.permute(1,2,0))
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Ground Truth")
plt.imshow(mask[0], cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("ResNet18")
plt.imshow(pred18[0], cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("ResNet34")
plt.imshow(pred34[0], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("comparison_resnet18_vs_resnet34.png")
plt.close()

# -----------------------
# Save separate images too
# -----------------------
plt.imsave("segmentation_resnet18.png", pred18[0], cmap="gray")
plt.imsave("segmentation_resnet34.png", pred34[0], cmap="gray")

print("Saved:")
print(" - comparison_resnet18_vs_resnet34.png")
print(" - segmentation_resnet18.png")
print(" - segmentation_resnet34.png")
