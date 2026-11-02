import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from vocab import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform

        self.imgs = []
        self.captions = []

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

                self.imgs.append(img_name)
                self.captions.append(caption)

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
        except:
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption)
