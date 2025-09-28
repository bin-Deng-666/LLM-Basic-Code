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

# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    
    # 创建RoPE层
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=512)
    
    # 创建模拟的查询和键张量
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # 创建位置ID
    position_ids = torch.arange(seq_len)
    
    print(f"原始查询形状: {q.shape}")
    print(f"原始键形状: {k.shape}")
    print(f"位置ID形状: {position_ids.shape}")
    
    # 应用旋转位置编码
    q_rotated, k_rotated = rope(q, k, position_ids)
    
    print(f"旋转后查询形状: {q_rotated.shape}")
    print(f"旋转后键形状: {k_rotated.shape}")
    
    # 测试不同序列长度
    seq_len2 = 256
    q2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    k2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    position_ids2 = torch.arange(seq_len2)
    
    q_rotated2, k_rotated2 = rope(q2, k2, position_ids2)
    print(f"长序列旋转后查询形状: {q_rotated2.shape}")