import torch
import torch.nn as nn
import math

# -------------------- 1. 位置编码 --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, hidden_dim]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.pe[:X.size(1)]  # X.shape: [B, L, D]

# -------------------- 2. 多头注意力基类 --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int):
        super().__init__()
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def _split_heads(self, X: torch.Tensor) -> torch.Tensor:
        B, L, _ = X.size()
        return X.view(B, L, self.head_num, self.head_dim).transpose(1, 2)  # [B, H, L, D]

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        Q = self._split_heads(self.Wq(Q))  # [B, H, Lq, D]
        K = self._split_heads(self.Wk(K))  # [B, H, Lk, D]
        V = self._split_heads(self.Wv(V))  # [B, H, Lv, D]

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, H, Lq, Lk]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [B, H, Lq, D]
        output = output.transpose(1, 2).contiguous().view(B, -1, self.head_num * self.head_dim)
        return self.out(output)

# -------------------- 3. Encoder 层 --------------------
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, head_num)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention
        attn_output = self.self_attn(X, X, X, src_mask)
        X = self.norm1(X + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(X)
        X = self.norm2(X + self.dropout(ffn_output))
        return X

# -------------------- 4. Decoder 层 --------------------
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, head_num)
        self.cross_attn = MultiHeadAttention(hidden_dim, head_num)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, enc_output: torch.Tensor, 
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Masked Self-Attention
        attn_output = self.self_attn(X, X, X, tgt_mask)
        X = self.norm1(X + self.dropout(attn_output))
        
        # Cross-Attention (Query来自Decoder，Key/Value来自Encoder)
        cross_output = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.norm2(X + self.dropout(cross_output))
        
        # FFN
        ffn_output = self.ffn(X)
        X = self.norm3(X + self.dropout(ffn_output))
        return X

# -------------------- 5. 完整 Transformer --------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 hidden_dim: int=512, head_num: int=8, ff_dim: int=2048, 
                 num_layers: int=6, dropout: float=0.1):
        super().__init__()
        # Encoder
        self.enc_embed = nn.Embedding(src_vocab_size, hidden_dim)
        self.enc_pos = PositionalEncoding(hidden_dim)
        self.enc_layers = nn.ModuleList([EncoderLayer(hidden_dim, head_num, ff_dim, dropout) 
                                       for _ in range(num_layers)])
        
        # Decoder
        self.dec_embed = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.dec_pos = PositionalEncoding(hidden_dim)
        self.dec_layers = nn.ModuleList([DecoderLayer(hidden_dim, head_num, ff_dim, dropout) 
                                       for _ in range(num_layers)])
        
        # Output
        self.proj = nn.Linear(hidden_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        X = self.dropout(self.enc_pos(self.enc_embed(src_ids)))
        for layer in self.enc_layers:
            X = layer(X, src_mask)
        return X

    def decode(self, tgt_ids: torch.Tensor, enc_output: torch.Tensor, 
               tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        X = self.dropout(self.dec_pos(self.dec_embed(tgt_ids)))
        for layer in self.dec_layers:
            X = layer(X, enc_output, tgt_mask, src_mask)
        return X

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        enc_output = self.encode(src_ids, src_mask)
        dec_output = self.decode(tgt_ids, enc_output, tgt_mask, src_mask)
        return self.proj(dec_output)

# -------------------- 6. 生成式推理示例 --------------------
def generate(model: Transformer, src_ids: torch.Tensor, max_len: int=20):
    model.eval()
    B = src_ids.size(0)
    device = src_ids.device
    
    # 编码器输出
    enc_output = model.encode(src_ids, None)
    
    # 初始化解码输入（假设起始符为0）
    tgt_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
    
    for _ in range(max_len):
        # 生成因果掩码
        L = tgt_ids.size(1)
        tgt_mask = torch.tril(torch.ones(L, L, device=device)).bool()
        
        # 解码
        logits = model(tgt_ids, enc_output, tgt_mask=tgt_mask)[:, -1, :]
        next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_ids], dim=1)
        
        # 遇到终止符停止（假设终止符为1）
        if (next_ids == 1).all():
            break
            
    return tgt_ids

# -------------------- 7. 测试用例 --------------------
if __name__ == "__main__":
    # 参数配置
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    hidden_dim = 512
    batch_size = 32
    seq_len = 64

    # 初始化模型
    model = Transformer(src_vocab_size, tgt_vocab_size, hidden_dim)

    # 生成测试数据
    src_ids = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    # 前向传播
    output = model(src_ids, tgt_ids)
    print(f"输出维度: {output.shape}")  # 应为 [32, 64, 8000]

    # 生成测试
    generated = generate(model, src_ids[:, :10])  # 输入前10个token
    print(f"生成序列维度: {generated.shape}")  # 如 [32, 20]