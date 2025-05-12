import torch
import torch.nn as nn

class KL(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, p_logits, q_logits):
        # 计算softmax
        p = torch.softmax(p_logits)
        q = torch.softmax(q_logits)
        # KL散度核心公式
        kl_div = (torch.log(p) - torch.log(q)) * p
        # 求和
        return torch.sum(kl_div, dim=-1)