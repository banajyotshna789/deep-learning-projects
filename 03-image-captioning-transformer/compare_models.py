import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from decoder_lstm import DecoderLSTM
from decoder_transformer import DecoderTransformer
from utils import FlickrDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Paths
# -------------------------
PROJECT2_DIR = "../dl_captioning"
PROJECT3_DIR = "."

LSTM_ENCODER_PATH = os.path.join(PROJECT2_DIR, "encoder.pth")
LSTM_DECODER_PATH = os.path.join(PROJECT2_DIR, "decoder.pth")

TRANS_CHECKPOINT_PATH = os.path.join(PROJECT3_DIR, "checkpoint_transformer.pth")

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# -------------------------
# Dataset / vocab / references
# -------------------------
dataset = FlickrDataset(
    root_dir="data/Flicker8k_Dataset",
    captions_file="data/Flickr8k.token.txt",
    transform=transform
)

vocab = dataset.vocab
start_idx = vocab.stoi["<start>"]
end_idx = vocab.stoi["<end>"]

def load_reference_captions(captions_file):
    refs = {}
    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_part, caption = line.split("\t", 1)
            img_name = img_part.split("#")[0]
            if img_name.endswith(".1"):
                img_name = img_name[:-2]
            refs.setdefault(img_name, []).append(caption)
    return refs

reference_captions = load_reference_captions("data/Flickr8k.token.txt")

# -------------------------
# LSTM model
# -------------------------
embed_size = 256
hidden_size = 256
vocab_size = len(vocab)

lstm_encoder = Encoder(embed_size).to(device)
lstm_decoder = DecoderLSTM(
    embed_size=embed_size,
    hidden_size=hidden_size,
    vocab_size=vocab_size,
    num_layers=1
).to(device)

lstm_encoder.load_state_dict(torch.load(LSTM_ENCODER_PATH, map_location=device))
lstm_decoder.load_state_dict(torch.load(LSTM_DECODER_PATH, map_location=device))

lstm_encoder.eval()
lstm_decoder.eval()

# -------------------------
# Transformer model
# -------------------------
trans_encoder = Encoder(embed_size).to(device)
trans_decoder = DecoderTransformer(
    embed_size=embed_size,
    vocab_size=vocab_size,
    num_heads=4,
    num_layers=2
).to(device)

checkpoint = torch.load(TRANS_CHECKPOINT_PATH, map_location=device)
trans_encoder.load_state_dict(checkpoint["encoder_state_dict"])
trans_decoder.load_state_dict(checkpoint["decoder_state_dict"])

trans_encoder.eval()
trans_decoder.eval()

# -------------------------
# Caption generation helpers
# -------------------------
def generate_lstm_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = lstm_encoder(image)

        caption_indices = [start_idx]
        for _ in range(max_len):
            captions_tensor = torch.tensor(caption_indices, dtype=torch.long).unsqueeze(0).to(device)
            outputs = lstm_decoder(features, captions_tensor)
            next_token = outputs[:, -1, :].argmax(dim=1).item()
            caption_indices.append(next_token)
            if next_token == end_idx and len(caption_indices) > 2:
                break

    words = []
    for idx in caption_indices[1:]:
        word = vocab.itos.get(idx, "<unk>")
        if word == "<end>":
            break
        if word not in ["<pad>", "<start>"]:
            words.append(word)

    return " ".join(words) if words else "[EMPTY CAPTION]"


def generate_transformer_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = trans_encoder(image)

        caption_indices = [start_idx]
        for _ in range(max_len):
            captions_tensor = torch.tensor(caption_indices, dtype=torch.long).unsqueeze(0).to(device)
            outputs = trans_decoder(features, captions_tensor)
            next_token = outputs[:, -1, :].argmax(dim=1).item()
            caption_indices.append(next_token)
            if next_token == end_idx and len(caption_indices) > 2:
                break

    words = []
    for idx in caption_indices[1:]:
        word = vocab.itos.get(idx, "<unk>")
        if word == "<end>":
            break
        if word not in ["<pad>", "<start>"]:
            words.append(word)

    return " ".join(words) if words else "[EMPTY CAPTION]"

# -------------------------
# Figure output
# -------------------------
def save_comparison_figure(image_path, reference_caption, lstm_caption, transformer_caption, save_path):
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Image", fontsize=12)

    axes[1].axis("off")
    text = (
        f"Reference:\n{reference_caption}\n\n"
        f"LSTM Caption:\n{lstm_caption}\n\n"
        f"Transformer Caption:\n{transformer_caption}"
    )
    axes[1].text(
        0.0, 1.0, text,
        fontsize=11,
        va="top",
        wrap=True
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    images = [
        "1000268201_693b08cb0e.jpg",
        "1001773457_577c3a7d70.jpg",
        "1002674143_1b742ab4b8.jpg",
        "1003163366_44323f5815.jpg",
        "1007129816_e794419615.jpg"
    ]

    os.makedirs("comparison_results", exist_ok=True)

    with open("comparison_results/comparison_results.txt", "w") as f:
        for img in images:
            image_path = os.path.join("data/Flicker8k_Dataset", img)

            refs = reference_captions.get(img, [])
            reference = refs[0] if refs else "[No reference found]"

            lstm_caption = generate_lstm_caption(image_path)
            transformer_caption = generate_transformer_caption(image_path)

            print("\n====================")
            print(f"Image: {img}")
            print(f"Reference: {reference}")
            print(f"LSTM: {lstm_caption}")
            print(f"Transformer: {transformer_caption}")

            f.write("========================================\n")
            f.write(f"Image: {img}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"LSTM: {lstm_caption}\n")
            f.write(f"Transformer: {transformer_caption}\n\n")

            save_path = os.path.join("comparison_results", img.replace(".jpg", ".png"))
            save_comparison_figure(image_path, reference, lstm_caption, transformer_caption, save_path)
