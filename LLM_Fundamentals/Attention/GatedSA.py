import torch.nn as nn
import torch

class GatedSelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, X, mask=None):
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.hidden_size)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, value)

        gate = self.gate(X)
        attn_output = attn_output * gate

        attn_output = self.output(attn_output)
        return attn_output