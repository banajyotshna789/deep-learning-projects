import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=4, num_layers=2):
        super(DecoderTransformer, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def generate_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, features, captions):
        # captions: (batch, seq_len)
        embeddings = self.embed(captions)
        embeddings = self.positional_encoding(embeddings)

        # features → (batch, 1, embed)
        features = features.unsqueeze(1)

        tgt_mask = self.generate_mask(captions.size(1)).to(captions.device)

        outputs = self.transformer(
            tgt=embeddings,
            memory=features,
            tgt_mask=tgt_mask
        )

        outputs = self.fc(outputs)
        return outputs
