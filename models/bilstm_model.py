import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM-based language model.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        embed_dim (int): Dimension of the embedding vectors.
        hidden_dim (int): Size of the LSTM hidden state.
        num_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout rate applied between LSTM layers.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        """
        Forward pass to compute vocabulary logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length)

        Returns:
            Tensor: Logits of shape (batch_size, sequence_length, vocab_size)
        """
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)

    def generate(self, prompt, tokenizer, device, max_len=50, temperature=1.0, eos_token=None):
        """
        Autoregressive token generation given a text prompt.

        Args:
            prompt (str): Initial input string.
            tokenizer: Tokenizer with encode/decode methods.
            device: Torch device (CPU or CUDA).
            max_len (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature for logits.
            eos_token (int, optional): Token ID to end generation early.

        Returns:
            str: Generated text output.
        """
        self.eval()
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        generated = ids[:]

        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(input_ids)[0, -1] / temperature
                next_token_id = torch.argmax(logits).item()
                if eos_token is not None and next_token_id == eos_token:
                    break
                generated.append(next_token_id)
                input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

        return tokenizer.decode(generated)
