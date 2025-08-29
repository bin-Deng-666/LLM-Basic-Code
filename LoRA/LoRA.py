import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.weight = weight
        self.weight.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha

        input_size, output_size = weight.size()
        self.lora_A = nn.Parameter(torch.empty(input_size, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.empty(rank, output_size))
        nn.init.zeros_(self.lora_B)
        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = torch.mm(X, self.weight)
        lora_path = torch.mm(torch.mm(X, self.lora_A), self.lora_B) * (self.alpha / self.rank)
        return y + lora_path