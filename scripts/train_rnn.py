import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import json
import random
import numpy as np
import os

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe_tokenizer.model")
EOS_ID = sp.piece_to_id("</s>")
print("Tokenizer loaded. Vocab size:", sp.get_piece_size())

# Hyperparameters
VOCAB_SIZE = 10000
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
SEQ_LEN = 32
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3

# Dataset class
class TextDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        all_text = [" ".join([entry["prompt"], entry["completion"]]) for entry in lines]
        tokens = sp.encode(" ".join(all_text))

        self.inputs = [tokens[i:i+SEQ_LEN] for i in range(len(tokens) - SEQ_LEN)]
        self.labels = [tokens[i+1:i+SEQ_LEN+1] for i in range(len(tokens) - SEQ_LEN)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])

# RNN model definition
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

    def generate(self, prompt, tokenizer, device, max_len=20, temperature=1.0, eos_token=None):
        self.eval()
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        generated = tokens[:]

        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                next_id = torch.argmax(next_token_logits).item()

                if eos_token is not None and next_id == eos_token:
                    break

                generated.append(next_id)
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)

        return tokenizer.decode(generated)

# Load data
dataset = TextDataset("train.jsonl")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Dataset ready: {len(dataset)} samples")

# Initialize model
model = RNNModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {avg_loss:.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/rnn_model.pt")
print("Model saved to checkpoints/rnn_model.pt")

# Generate from prompt
prompt = "Which do you prefer? Dogs or cats?"
print("Prompt:", prompt)
print("RNN Response:", model.generate(prompt, sp, device, max_len=30, temperature=1.0, eos_token=EOS_ID))

