import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float):
        super().__init__()
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        var = torch.var(x, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps) # (batch_size, seq_len, hidden_dim)
        output = self.gamma * x_normalized + self.beta # (batch_size, seq_len, hidden_dim)
        return output
        