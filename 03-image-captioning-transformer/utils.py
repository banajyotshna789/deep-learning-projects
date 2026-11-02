import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from vocab import Vocabulary

# Allow loading slightly corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform

        self.imgs = []
        self.captions = []

        print("Loading captions and filtering dataset...")

        with open(captions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                img_name = parts[0].split('#')[0]
                caption = parts[1]

                # Fix filenames like .jpg.1
                if img_name.endswith(".1"):
                    img_name = img_name[:-2]

                img_path = os.path.join(self.root_dir, img_name)

                # Keep only valid images
                if os.path.exists(img_path):
                    self.imgs.append(img_name)
                    self.captions.append(caption)
                else:
                    print(f"Skipping missing image: {img_path}")

        print(f"Final dataset size after filtering: {len(self.imgs)}")

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {img_path}")
            raise e

        if self.transform:
            image = self.transform(image)

        # Convert caption to numerical tokens
        caption_tokens = [self.vocab.stoi["<start>"]]
        caption_tokens += self.vocab.numericalize(caption)
        caption_tokens.append(self.vocab.stoi["<end>"])

        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        return image, caption_tensor
