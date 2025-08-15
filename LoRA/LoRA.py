import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.weight = weight
        self.rank = rank
        self.alpha = alpha
        self.weight.requires_grad_(False)

        out_features, in_features = weight.size()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = F.linear(X, self.weight)                 # 原始前向
        lora_path = (F.linear(F.linear(X, self.lora_A), self.lora_B)
                     * (self.alpha / self.rank))     # 缩放
        return y + lora_path