import torch
from torch.utils.data import Dataset
import sentencepiece as spm
class TextDataset(Dataset):
    def __init__(self, filepath, sp_model_path, seq_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        token_ids = self.sp.encode(text, out_type=int)
        self.seq_len = seq_len
        self.inputs = []
        self.targets = []
        for i in range(0, len(token_ids) - seq_len):
            seq = token_ids[i:i + seq_len]
            target = token_ids[i + 1:i + 1 + seq_len]
            self.inputs.append(torch.tensor(seq, dtype=torch.long))
            self.targets.append(torch.tensor(target, dtype=torch.long))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
