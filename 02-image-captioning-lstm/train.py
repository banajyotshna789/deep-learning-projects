import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from encoder import Encoder
from decoder_lstm import DecoderLSTM
from utils import FlickrDataset

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1

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
decoder = DecoderLSTM(
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    vocab_size=len(dataset.vocab),
    num_layers=NUM_LAYERS
).to(DEVICE)

# =========================
# Loss and optimizer
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

# =========================
# Training loop
# =========================
print("Starting training...", flush=True)

for epoch in range(NUM_EPOCHS):
    encoder.train()
    decoder.train()

    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}", flush=True)

    for i, (images, captions) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        captions = captions.to(DEVICE, non_blocking=True)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)

        # Shift captions for teacher forcing loss
        targets = captions[:, 1:]
        outputs = outputs[:, :-1, :]

        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Step [{i}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = running_loss / max(1, len(train_loader))
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}", flush=True)

# =========================
# Save checkpoints
# =========================
print("Saving models...", flush=True)
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
print("Training complete. Models saved as encoder.pth and decoder.pth", flush=True)

