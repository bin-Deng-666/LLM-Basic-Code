import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float):
        super().__init__()
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(hidden_dim)) # (hidden_dim,)
        
    def forward(self, x):
        x_squared = x.pow(2) # (batch_size, seq_len, hidden_dim)
        mean_squared = torch.mean(x_squared, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        rms = torch.sqrt(mean_squared + self.eps) # (batch_size, seq_len, 1)
        x_normalized = x / rms # (batch_size, seq_len, hidden_dim)
        output = x_normalized * self.gamma # (batch_size, seq_len, hidden_dim)
        return output