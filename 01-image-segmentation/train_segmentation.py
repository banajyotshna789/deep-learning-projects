import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# =============================
# Configuration
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
IMAGE_SIZE = 128

BACKBONES = ["resnet18", "resnet34"]

# =============================
# Transforms
# =============================
image_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.PILToTensor(),
])

# =============================
# Dataset
# =============================
class PetSegmentation(OxfordIIITPet):
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        image = image_transform(image)
        mask = mask_transform(mask).float()

        # Binary segmentation: pet vs background
        mask = (mask == 2).float()
        return image, mask

dataset = PetSegmentation(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=False
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =============================
# Metrics
# =============================
def dice_score(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# =============================
# Training Function
# =============================
def train_model(backbone):

    print("\n" + "="*50)
    print(f"Training U-Net with {backbone}")
    print("="*50)

    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_iou = 0
    final_dice = 0
    final_iou = 0

    for epoch in range(EPOCHS):

        # ---- Training ----
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)

                val_dice += dice_score(outputs, masks).item()
                val_iou += iou_score(outputs, masks).item()

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Val IoU: {val_iou:.4f}")
        print("-"*40)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"model_{backbone}.pth")

        final_dice = val_dice
        final_iou = val_iou

    print(f"\nBest IoU for {backbone}: {best_iou:.4f}")
    return final_dice, final_iou

# =============================
# Run Experiments
# =============================
results = {}

for backbone in BACKBONES:
    dice, iou = train_model(backbone)
    results[backbone] = (dice, iou)

# =============================
# Final Summary
# =============================
print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)

for backbone in results:
    print(f"{backbone} → Dice: {results[backbone][0]:.4f}, IoU: {results[backbone][1]:.4f}")
