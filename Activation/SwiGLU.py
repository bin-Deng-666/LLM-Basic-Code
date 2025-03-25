import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model  # 默认扩大4倍
        self.w = nn.Linear(d_model, hidden_dim, bias=False)
        self.v = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()  # Swish别名
        self.w_o = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_o(self.act(self.w(x)) * self.v(x))