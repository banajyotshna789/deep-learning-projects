import torch
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from decoder_transformer import DecoderTransformer
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
start_idx = vocab.stoi["<start>"]
end_idx = vocab.stoi["<end>"]

# =========================
# Model config
# =========================
embed_size = 256
vocab_size = len(vocab)
num_heads = 4
num_layers = 2

encoder = Encoder(embed_size).to(device)
decoder = DecoderTransformer(
    embed_size=embed_size,
    vocab_size=vocab_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

# =========================
# Load checkpoint
# =========================
checkpoint = torch.load("checkpoint_transformer.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

encoder.eval()
decoder.eval()

# =========================
# Caption generation
# =========================
def generate_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)

        caption_indices = [start_idx]

        for _ in range(max_len):
            captions_tensor = torch.tensor(
                caption_indices,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            outputs = decoder(features, captions_tensor)
            next_token_logits = outputs[:, -1, :]
            predicted_idx = next_token_logits.argmax(dim=1).item()

            caption_indices.append(predicted_idx)

            if predicted_idx == end_idx:
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

    for img in images:
        path = f"data/Flicker8k_Dataset/{img}"
        caption = generate_caption(path)

        print("\n====================")
        print(f"Image: {img}")
        print(f"Caption: {caption}")
