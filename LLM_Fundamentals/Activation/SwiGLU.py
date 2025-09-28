import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4 * dim  # 默认扩大4倍
        self.w = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.SiLU()  # Swish别名
        self.w_o = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_o(self.act(self.w(x)) * self.v(x))