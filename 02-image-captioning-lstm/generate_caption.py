import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from decoder_lstm import DecoderLSTM
from utils import FlickrDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Transform
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
# Load dataset (for vocab)
# =========================
dataset = FlickrDataset(
    root_dir="data/Flicker8k_Dataset",
    captions_file="data/Flickr8k.token.txt",
    transform=transform
)

vocab = dataset.vocab

# =========================
# Model config
# =========================
embed_size = 256
hidden_size = 256
vocab_size = len(vocab)

encoder = Encoder(embed_size).to(device)
decoder = DecoderLSTM(embed_size, hidden_size, vocab_size).to(device)

# =========================
# Load trained weights
# =========================
encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()

# =========================
# Load reference captions
# =========================
def load_reference_captions(captions_file):
    refs = {}
    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_part, caption = line.split("\t", 1)
            image_name = image_part.split("#")[0]
            refs.setdefault(image_name, []).append(caption)
    return refs

reference_captions = load_reference_captions("data/Flickr8k.token.txt")

# =========================
# Caption generation
# =========================
def generate_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)

        start_idx = vocab.stoi["<start>"]
        end_idx = vocab.stoi["<end>"]

        caption_indices = [start_idx]

        for _ in range(max_len):
            captions_tensor = torch.tensor(
                caption_indices, dtype=torch.long
            ).unsqueeze(0).to(device)

            outputs = decoder(features, captions_tensor)
            next_token_logits = outputs[:, -1, :]
            predicted_idx = next_token_logits.argmax(dim=1).item()

            caption_indices.append(predicted_idx)

            if predicted_idx == end_idx and len(caption_indices) > 2:
                break

        words = []
        for idx in caption_indices[1:]:
            word = vocab.itos.get(idx, "<unk>")
            if word == "<end>":
                break
            if word not in ["<pad>", "<start>"]:
                words.append(word)

        if len(words) == 0:
            return "[EMPTY CAPTION]"

        return " ".join(words)

# =========================
# Save image + captions
# =========================
def save_result_figure(image_path, reference_caption, generated_caption, save_path):
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"Reference: {reference_caption}\nGenerated: {generated_caption}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

# =========================
# Example run
# =========================
if __name__ == "__main__":
    images = [
        "1000268201_693b08cb0e.jpg",
        "1001773457_577c3a7d70.jpg",
        "1002674143_1b742ab4b8.jpg",
        "1003163366_44323f5815.jpg",
        "1007129816_e794419615.jpg"
    ]

    os.makedirs("results", exist_ok=True)

    with open("results/results.txt", "w") as f:
        for img in images:
            image_path = f"data/Flicker8k_Dataset/{img}"
            generated = generate_caption(image_path)

            refs = reference_captions.get(img, [])
            reference = refs[0] if refs else "[No reference caption found]"

            print("\n====================")
            print(f"Image: {img}")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")

            f.write("========================================\n")
            f.write(f"Image: {img}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"Generated: {generated}\n\n")

            save_path = os.path.join("results", img.replace(".jpg", ".png"))
            save_result_figure(image_path, reference, generated, save_path)
