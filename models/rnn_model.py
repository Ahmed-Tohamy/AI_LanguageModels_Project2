import torch

import torch.nn as nn



class RNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1):

        super(RNNModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)



    def forward(self, x):

        x = self.embed(x)

        output, _ = self.rnn(x)

        logits = self.fc(output)

        return logits

