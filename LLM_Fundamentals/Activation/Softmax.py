import torch
import torch.nn as nn

class SafeSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        max_val = torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x - max_val)
        sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
        return exp_x / sum_exp