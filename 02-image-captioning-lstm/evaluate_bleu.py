import os
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from decoder_lstm import DecoderLSTM
from utils import FlickrDataset

# If needed once:
# nltk.download("punkt")

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
# Dataset / vocab
# =========================
dataset = FlickrDataset(
    root_dir="data/Flicker8k_Dataset",
    captions_file="data/Flickr8k.token.txt",
    transform=transform
)
vocab = dataset.vocab

# =========================
# Models
# =========================
embed_size = 256
hidden_size = 256
vocab_size = len(vocab)

encoder = Encoder(embed_size).to(device)
decoder = DecoderLSTM(embed_size, hidden_size, vocab_size).to(device)

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
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)

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
# BLEU evaluation
# =========================
def tokenize_caption(text):
    return nltk.word_tokenize(text.lower())

if __name__ == "__main__":
    images = [
        "1000268201_693b08cb0e.jpg",
        "1001773457_577c3a7d70.jpg",
        "1002674143_1b742ab4b8.jpg",
        "1003163366_44323f5815.jpg",
        "1007129816_e794419615.jpg"
    ]

    smooth = SmoothingFunction().method4

    references = []
    candidates = []
    per_image_scores = []

    os.makedirs("results", exist_ok=True)
    out_file = "results/bleu_results.txt"

    with open(out_file, "w") as f:
        for img in images:
            image_path = f"data/Flicker8k_Dataset/{img}"
            generated = generate_caption(image_path)

            refs = reference_captions.get(img, [])
            ref_tokens = [tokenize_caption(r) for r in refs]
            cand_tokens = tokenize_caption(generated)

            references.append(ref_tokens)
            candidates.append(cand_tokens)

            bleu1 = corpus_bleu([ref_tokens], [cand_tokens], weights=(1, 0, 0, 0), smoothing_function=smooth)
            bleu2 = corpus_bleu([ref_tokens], [cand_tokens], weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            bleu3 = corpus_bleu([ref_tokens], [cand_tokens], weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
            bleu4 = corpus_bleu([ref_tokens], [cand_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

            per_image_scores.append((bleu1, bleu2, bleu3, bleu4))

            print("\n====================")
            print(f"Image: {img}")
            print(f"Reference: {refs[0] if refs else '[No reference found]'}")
            print(f"Generated: {generated}")
            print(f"BLEU-1: {bleu1:.4f}  BLEU-2: {bleu2:.4f}  BLEU-3: {bleu3:.4f}  BLEU-4: {bleu4:.4f}")

            f.write("========================================\n")
            f.write(f"Image: {img}\n")
            f.write(f"Reference: {refs[0] if refs else '[No reference found]'}\n")
            f.write(f"Generated: {generated}\n")
            f.write(f"BLEU-1: {bleu1:.4f}\n")
            f.write(f"BLEU-2: {bleu2:.4f}\n")
            f.write(f"BLEU-3: {bleu3:.4f}\n")
            f.write(f"BLEU-4: {bleu4:.4f}\n\n")

        corpus_bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=smooth)
        corpus_bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        corpus_bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
        corpus_bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        print("\n====================")
        print("Corpus BLEU Scores")
        print(f"BLEU-1: {corpus_bleu1:.4f}")
        print(f"BLEU-2: {corpus_bleu2:.4f}")
        print(f"BLEU-3: {corpus_bleu3:.4f}")
        print(f"BLEU-4: {corpus_bleu4:.4f}")

        f.write("========================================\n")
        f.write("Corpus BLEU Scores\n")
        f.write(f"BLEU-1: {corpus_bleu1:.4f}\n")
        f.write(f"BLEU-2: {corpus_bleu2:.4f}\n")
        f.write(f"BLEU-3: {corpus_bleu3:.4f}\n")
        f.write(f"BLEU-4: {corpus_bleu4:.4f}\n")
