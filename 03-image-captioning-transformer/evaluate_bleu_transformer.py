import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image
from torchvision import transforms

from encoder import Encoder
from decoder_transformer import DecoderTransformer
from utils import FlickrDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Dataset / vocab
# -------------------------
dataset = FlickrDataset(
    root_dir="data/Flicker8k_Dataset",
    captions_file="data/Flickr8k.token.txt",
    transform=transform
)

vocab = dataset.vocab

# -------------------------
# Load model (CHECKPOINT)
# -------------------------
embed_size = 256
vocab_size = len(vocab)

encoder = Encoder(embed_size).to(device)
decoder = DecoderTransformer(embed_size, vocab_size).to(device)

checkpoint = torch.load("checkpoint_transformer.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

encoder.eval()
decoder.eval()

start_idx = vocab.stoi["<start>"]
end_idx = vocab.stoi["<end>"]

# -------------------------
# Caption generation
# -------------------------
def generate_caption(image_path, max_len=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        caption = [start_idx]

        for _ in range(max_len):
            cap_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            outputs = decoder(features, cap_tensor)
            next_token = outputs[:, -1, :].argmax(dim=1).item()
            caption.append(next_token)

            if next_token == end_idx:
                break

    words = []
    for idx in caption[1:]:
        word = vocab.itos.get(idx, "<unk>")
        if word == "<end>":
            break
        if word not in ["<pad>", "<start>"]:
            words.append(word)

    return words

# -------------------------
# Load references
# -------------------------
def load_refs(file):
    refs = {}
    with open(file, "r") as f:
        for line in f:
            img_part, cap = line.strip().split("\t")
            img = img_part.split("#")[0]
            refs.setdefault(img, []).append(nltk.word_tokenize(cap.lower()))
    return refs

refs_dict = load_refs("data/Flickr8k.token.txt")

# -------------------------
# Evaluate
# -------------------------
images = list(refs_dict.keys())[:100]   # use 100 images

references = []
candidates = []

smooth = SmoothingFunction().method4

for img in images:
    path = f"data/Flicker8k_Dataset/{img}"
    pred = generate_caption(path)

    references.append(refs_dict[img])
    candidates.append(pred)

bleu1 = corpus_bleu(references, candidates, weights=(1,0,0,0), smoothing_function=smooth)
bleu4 = corpus_bleu(references, candidates, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

print("\n=== Transformer BLEU ===")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
