import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入的一半维度旋转，作为 RoPE 的辅助函数"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # 预计算 cos, sin
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        cos = self.cos_cached[:, :, position_ids, :]
        sin = self.sin_cached[:, :, position_ids, :]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed