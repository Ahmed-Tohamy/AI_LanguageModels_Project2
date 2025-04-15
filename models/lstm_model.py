
import torch

import torch.nn as nn



class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1):

        super(LSTMModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)



    def forward(self, x):

        x = self.embed(x)

        output, _ = self.lstm(x)

        logits = self.fc(output)

        return logits


