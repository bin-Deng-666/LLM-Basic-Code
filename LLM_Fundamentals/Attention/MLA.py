import torch.nn as nn
import torch

# 假设已经定义了RMSNorm和Rope
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
    
    def forward(self, X):
        return None
        

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
    def forward(self, query, key):
        return None
        

class DeepSeekConfig:
    hidden_dim: int
    num_heads: int
    dropout_rate: float
    # query
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    # key
    kv_lora_rank: int
    # value
    v_head_dim: int
    

class MLA(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        # 初始化记录
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        # query
        self.q_down_proj = nn.Linear(self.hidden_dim, self.q_lora_rank)
        self.q_down_norm = RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        
        # key和value
        self.kv_down_proj = nn.Linear(self.hidden_dim, self.qk_rope_head_dim + self.kv_lora_rank)
        self.kv_down_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        
        # rope
        self.rope = RotaryEmbedding(self.qk_rope_head_dim)
        
        # others
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_dim)
    
    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()
        
        """query部分"""
        q = self.q_down_proj(X)
        q = self.q_down_norm(q)
        q = self.q_up_proj(q) 
        # (batch_size, seq_len, num_heads * (qk_nope_head_dim + qk_rope_head_dim)) -> (batch_size, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        """key和value部分"""
        c_kv = self.kv_down_proj(X)
        k_rope, c_kv = torch.split(c_kv, [self.qk_rope_head_dim, self.kv_lora_rank], dim=-1)
        # (batch_size, seq_len, qk_rope_head_dim) -> (batch_size, 1, seq_len, qk_rope_head_dim)
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        c_kv = self.kv_down_norm(c_kv)
        kv = self.kv_up_proj(c_kv)
        # (batch_size, seq_len, num_heads * (qk_nope_head_dim + v_head_dim)) -> (batch_size, num_heads, seq_len, qk_nope_head_dim + v_head_dim)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        """位置编码"""
        q_rope, k_rope = self.rope(q_rope, k_rope)
        
        """拼接"""
        # (batch_size, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        query = torch.concat([q_nope, q_rope], dim=-1)
        # (batch_size, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        key = torch.concat([k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)
    
        """后续操作"""
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / torch.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        if attention_mask is not None:
            attn_scores = torch.masked_fill(attn_scores, attention_mask == 0, float('-inf'))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, value)
        # (batch_size, num_heads, seq_len, v_head_dim) -> (batch_size, seq_len, num_heads * v_head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.output_proj(attn_output)
        
        return output
        