import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from encoder import Encoder
from decoder_transformer import DecoderTransformer
from utils import FlickrDataset

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EXTRA_EPOCHS = 10
LEARNING_RATE = 3e-4
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 2

CHECKPOINT_PATH = "checkpoint_transformer.pth"
LAST_ENCODER_PATH = "encoder_transformer.pth"
LAST_DECODER_PATH = "decoder_transformer.pth"
BEST_ENCODER_PATH = "encoder_transformer_best.pth"
BEST_DECODER_PATH = "decoder_transformer_best.pth"

# =========================
# Image transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# =========================
# Dataset and DataLoader
# =========================
print("Loading dataset...", flush=True)

dataset = FlickrDataset(
    root_dir="data/Flicker8k_Dataset",
    captions_file="data/Flickr8k.token.txt",
    transform=transform
)

pad_idx = dataset.vocab.stoi["<pad>"]

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return images, captions

train_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

print(f"Dataset loaded. Size: {len(dataset)}", flush=True)
print(f"Vocabulary size: {len(dataset.vocab)}", flush=True)

# =========================
# Models
# =========================
encoder = Encoder(embed_size=EMBED_SIZE).to(DEVICE)
decoder = DecoderTransformer(
    embed_size=EMBED_SIZE,
    vocab_size=len(dataset.vocab),
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(DEVICE)

# =========================
# Loss and optimizer
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

# =========================
# Checkpoint loading
# =========================
start_epoch = 0
best_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...", flush=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch}", flush=True)
else:
    print("No checkpoint found. Starting fresh training.", flush=True)

# =========================
# Helper: save checkpoint
# =========================
def save_checkpoint(epoch, avg_loss, best_loss_value):
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss_value,
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
        },
        CHECKPOINT_PATH
    )

# =========================
# Training loop
# =========================
print("Starting training...", flush=True)

end_epoch = start_epoch + EXTRA_EPOCHS

for epoch in range(start_epoch, end_epoch):
    encoder.train()
    decoder.train()

    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{end_epoch}", flush=True)

    for i, (images, captions) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        captions = captions.to(DEVICE, non_blocking=True)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)

        # Teacher forcing shift
        targets = captions[:, 1:]
        outputs = outputs[:, :-1, :]

        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{end_epoch}] "
                f"Step [{i}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = running_loss / max(1, len(train_loader))
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}", flush=True)

    # Save last checkpoint every epoch
    save_checkpoint(epoch, avg_loss, best_loss)
    torch.save(encoder.state_dict(), LAST_ENCODER_PATH)
    torch.save(decoder.state_dict(), LAST_DECODER_PATH)

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(encoder.state_dict(), BEST_ENCODER_PATH)
        torch.save(decoder.state_dict(), BEST_DECODER_PATH)
        print(f"Saved new best model with loss {best_loss:.4f}", flush=True)

print("Training complete.", flush=True)
print(f"Last weights saved to {LAST_ENCODER_PATH} and {LAST_DECODER_PATH}", flush=True)
print(f"Best weights saved to {BEST_ENCODER_PATH} and {BEST_DECODER_PATH}", flush=True)
