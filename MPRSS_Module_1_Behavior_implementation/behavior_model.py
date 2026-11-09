"""
behavior_model.py - the Module 1 network (needs PyTorch).

One definition, imported by BOTH train_final.py and the live service, so the saved
weights always load into exactly the architecture they were trained with.
(Layer names - emb, lstm, drop, out - match the SeqClassifier used in train_lstm.py.)
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SeqClassifier(nn.Module):
    """Embedding -> LSTM -> last hidden state -> 1 score (benign vs malware)."""

    def __init__(self, vocab_size, emb_dim=16, hidden=32, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)   # id 0 = padding
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(h[-1])).squeeze(1)