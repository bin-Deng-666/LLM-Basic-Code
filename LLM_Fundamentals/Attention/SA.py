import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, X, attention_mask=None):
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        
        attn_output = torch.matmul(attn_scores, V)
        output = self.output(attn_output)
        
        return output