import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from dataset import TextDataset
from models.transformer import TransformerModel

# === Constants ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "bpe_tokenizer.model"
TEST_PATH = "test.jsonl"
VOCAB_SIZE = 10000
BATCH_SIZE = 64

# === Load Tokenizer ===
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(TOKENIZER_PATH)
EOS_ID = tokenizer.piece_to_id("</s>")
print(f"Loaded tokenizer with vocab size {tokenizer.get_piece_size()}")

# === Load Test Dataset ===
test_dataset = TextDataset(TEST_PATH, TOKENIZER_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"Loaded test dataset: {len(test_dataset)} samples")

# === Metrics ===
def compute_perplexity(model, data_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0)
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)

def compute_bleu(model, data_loader, tokenizer, device, max_len=50):
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for x, y in data_loader:
        x = x.to(device)
        for i in range(x.size(0)):
            input_ids = x[i].tolist()
            ref_ids = y[i].tolist()
            prompt = tokenizer.decode(input_ids[:10])
            ref = tokenizer.decode(ref_ids)
            gen = model.generate(prompt, tokenizer, device, max_len=max_len, temperature=1.0, eos_token=EOS_ID)
            score = sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)
            bleu_scores.append(score)
        break  # Only evaluate on one batch for BLEU
    return np.mean(bleu_scores)

# === RNN Model ===
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.rnn = nn.RNN(64, 128, 1, batch_first=True)
        self.fc = nn.Linear(128, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
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

# === LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 256)
        self.lstm = nn.LSTM(256, 512, 2, batch_first=True)
        self.fc = nn.Linear(512, VOCAB_SIZE)

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
                next_token_id = torch.argmax(logits).item()
                if eos_token is not None and next_token_id == eos_token:
                    break
                generated.append(next_token_id)
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)
        return tokenizer.decode(generated)

# === BiLSTM Model ===
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 256)
        self.lstm = nn.LSTM(256, 512, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512 * 2, VOCAB_SIZE)

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
                next_token_id = torch.argmax(logits).item()
                if eos_token is not None and next_token_id == eos_token:
                    break
                generated.append(next_token_id)
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)
        return tokenizer.decode(generated)

# === Evaluation Function ===
def eval_model(name, model_class, ckpt_path, tokenizer, device, init_args=None, subkey=None):
    print(f"\nEvaluating {name}...")
    model = model_class(**init_args).to(device) if init_args else model_class().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict[subkey] if subkey else state_dict)
    ppl = compute_perplexity(model, test_loader)
    bleu = compute_bleu(model, test_loader, tokenizer, device)
    print(f"{name} Perplexity: {ppl:.2f}")
    print(f"{name} BLEU Score: {bleu:.4f}")
    print("Sample Generations:")
    prompts = [
        "Which do you prefer? Dogs or cats?",
        "The sun rises over the mountain",
        "In the future, AI will",
        "Explain photosynthesis in simple terms.",
        "She opened the door and found"
    ]
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print(f"{name} Response: {model.generate(prompt, tokenizer, device, max_len=50, temperature=1.0, eos_token=EOS_ID)}")

# === Run All Models ===
if __name__ == "__main__":
    eval_model("RNN", RNNModel, "checkpoints/rnn_model.pt", tokenizer, DEVICE)
    eval_model("LSTM", LSTMModel, "checkpoints/lstm_model.pt", tokenizer, DEVICE)
    eval_model("BiLSTM", BiLSTMModel, "checkpoints/bilstm_model.pt", tokenizer, DEVICE, subkey="model_state")
    eval_model("Transformer", TransformerModel, "checkpoints/transformer_model.pt", tokenizer, DEVICE, init_args={
        "vocab_size": VOCAB_SIZE,
        "embed_dim": 256,
        "nhead": 8,
        "hidden_dim": 512,
        "num_layers": 4,
        "max_len": 512
    })
