import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Optional



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)



        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))



        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)



        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)



    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, nhead=8, hidden_dim=512, num_layers=4, max_len=512, dropout=0.1):

        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_dim = embed_dim



    def forward(self, x):

        """

        Args:

            x: Tensor of shape (batch_size, seq_len)

        Returns:

            logits: Tensor of shape (batch_size, seq_len, vocab_size)

        """

        x = self.embedding(x) * (self.embed_dim ** 0.5)

        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)

        logits = self.fc_out(output)

        return logits



    def generate(self, prompt: str, tokenizer, device: torch.device, max_len: int = 50, temperature: float = 1.0, eos_token: Optional[int] = None) -> str:

        """

        Autoregressively generate tokens from a prompt.



        Args:

            prompt (str): Text prompt

            tokenizer: SentencePieceProcessor

            device: torch.device

            max_len (int): Max generation length

            temperature (float): Sampling temperature

            eos_token (int, optional): End-of-sequence token id



        Returns:

            str: Decoded string

        """

        self.eval()

        ids = tokenizer.encode(prompt)

        input_ids = torch.tensor([ids], dtype=torch.long).to(device)



        with torch.no_grad():

            for _ in range(max_len):

                logits = self.forward(input_ids)

                next_token_logits = logits[0, -1, :] / temperature

                next_id = torch.argmax(next_token_logits).item()



                if eos_token is not None and next_id == eos_token:

                    break



                ids.append(next_id)

                input_ids = torch.tensor([ids], dtype=torch.long).to(device)



        return tokenizer.decode(ids)

