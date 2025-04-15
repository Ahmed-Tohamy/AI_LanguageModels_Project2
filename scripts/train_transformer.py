# train_transformer_full.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
import numpy as np
import random
import os
from tqdm import tqdm
from dataset import TextDataset
from models.transformer import TransformerModel

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()
print("Using device:", device)

# Load tokenizer
tokenizer_path = "bpe_tokenizer.model"
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
EOS_ID = sp.piece_to_id("</s>")
print("Tokenizer loaded with vocab size:", sp.get_piece_size())

# Hyperparameters
VOCAB_SIZE = 10000
BATCH_SIZE = 128
EPOCHS = 30
LR = 5e-4
MAX_LEN = 512
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 4
NHEAD = 8
PATIENCE = 3

# Load data
train_dataset = TextDataset("train.jsonl", tokenizer_path, seq_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Loaded {len(train_dataset)} training samples.")

# Initialize model
model = TransformerModel(VOCAB_SIZE, EMBED_DIM, NHEAD, HIDDEN_DIM, NUM_LAYERS, MAX_LEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

ckpt_path = "checkpoints/transformer_checkpoint.pt"
start_epoch = 0
best_loss = float("inf")
no_improve = 0

# Load checkpoint if exists
if os.path.exists(ckpt_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    no_improve = checkpoint["epochs_no_improve"]

# Training loop
print("Starting Transformer training...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, _ = batch
        x = x.to(device)
        input_seq, target_seq = x[:, :-1], x[:, 1:]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(input_seq)
            loss = criterion(logits.view(-1, VOCAB_SIZE), target_seq.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(model.state_dict(), "checkpoints/transformer_model.pt")
        print("Best model saved.")
    else:
        no_improve += 1
        print(f"No improvement. Patience: {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save training state
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "epochs_no_improve": no_improve
    }, ckpt_path)
