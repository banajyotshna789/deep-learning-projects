import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_dim=28*28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(img_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img, c], dim=1)
        return self.model(x)
