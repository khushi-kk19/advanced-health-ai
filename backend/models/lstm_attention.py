# backend/models/lstm_attention.py
import torch
import torch.nn as nn

class LSTMAttnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, attn_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        proj = self.proj(out)           # (batch, seq_len, hidden)
        attn_out, attn_weights = self.attn(proj, proj, proj)  # (batch, seq_len, hidden)
        pooled = attn_out.mean(dim=1)   # mean-pool across time
        out = self.fc(pooled)
        return out.squeeze(-1), attn_weights
