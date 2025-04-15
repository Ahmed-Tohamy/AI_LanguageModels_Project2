import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sentencepiece as spm
import numpy as np
import random
import os
from dataset import TextDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe_tokenizer.model")
EOS_ID = sp.piece_to_id("</s>")
print(f"Tokenizer loaded. Vocabulary size: {sp.get_piece_size()}")

# Hyperparameters
VOCAB_SIZE = 10000
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
PATIENCE = 4
VALID_SPLIT = 0.1

# Load and split dataset
full_dataset = TextDataset("train.jsonl", "bpe_tokenizer.model", seq_len=SEQ_LEN)
val_size = int(len(full_dataset) * VALID_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)

    def generate(self, prompt, tokenizer, device, max_len=50, temperature=1.0, eos_token=None):
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

# Initialize or load model
ckpt_path = "checkpoints/bilstm_model.pt"
model = BiLSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

start_epoch = 0
best_loss = float('inf')
patience_counter = 0

if os.path.exists(ckpt_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    patience_counter = checkpoint["patience"]

# Training loop
print("Starting BiLSTM training...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Validation Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    # Checkpointing
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "patience": patience_counter
        }, ckpt_path)
        print("Best model saved.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Prompt-based generation
prompt = "Which do you prefer? Dogs or cats?"
print("Prompt:", prompt)
print("BiLSTM Response:", model.generate(prompt, sp, device, max_len=50, temperature=1.0, eos_token=EOS_ID))
