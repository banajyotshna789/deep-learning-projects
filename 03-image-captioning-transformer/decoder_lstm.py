import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # remove last token (<end>)
        embeddings = self.embed(captions[:, :-1])

        # add image features as first input
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)

        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)

        return outputs
