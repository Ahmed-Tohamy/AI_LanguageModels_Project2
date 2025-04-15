import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
import numpy as np
import random
import os
from dataset import TextDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Select device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SentencePiece tokenizer
tokenizer_path = "bpe_tokenizer.model"
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)
EOS_ID = tokenizer.piece_to_id("</s>")
print(f"Tokenizer loaded with vocab size: {tokenizer.get_piece_size()}")

# Hyperparameters
VOCAB_SIZE = 10000
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
PATIENCE = 5

# Load dataset
train_dataset = TextDataset("train.jsonl", tokenizer_path, seq_len=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Training samples: {len(train_dataset)}")

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)

    def generate(self, prompt, tokenizer, device, max_len=50, temperature=1.0, eos_token=None):
        self.eval()
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        generated = ids[:]

        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(input_ids)[0, -1] / temperature
                next_id = torch.argmax(logits).item()
                if eos_token is not None and next_id == eos_token:
                    break
                generated.append(next_id)
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)

        return tokenizer.decode(generated)

# Initialize model, optimizer, loss, and scheduler
model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# Load checkpoint if it exists
checkpoint_path = "checkpoints/lstm_model.pt"
if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path))

# Training loop
print("Starting LSTM training...")
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Best model saved.")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Text generation
prompt = "Which do you prefer? Dogs or cats?"
print("Prompt:", prompt)
print("LSTM Response:", model.generate(prompt, tokenizer, device, temperature=1.0, eos_token=EOS_ID))
