from turtle import forward
import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.ouput_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, mask) -> torch.Tensor:
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.hidden_dim)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = torch.matmul(attn_scores, V)
        attn_output = self.ouput_proj(attn_output)
        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int, dropout_rate: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

        self.query = nn.Linear(hidden_dim, self.num_head * self.head_dim)
        self.key = nn.Linear(hidden_dim, self.num_head * self.head_dim)
        self.value = nn.Linear(hidden_dim, self.num_head * self.head_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, mask) -> torch.Tensor:
        bsz, seq_len, _ = X.shape

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        Q = Q.view(bsz, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = self.dropout(attn_scores)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_scores, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        attn_output = self.output_proj(attn_output)
        return attn_output


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, q_num_head: int, kv_num_head: int, dropout_rate: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // q_num_head
        self.q_num_head = q_num_head
        self.kv_num_head = kv_num_head
        self.num_per_group = q_num_head // kv_num_head

        self.query = nn.Linear(hidden_dim, q_num_head * self.head_dim)
        self.key = nn.Linear(hidden_dim, kv_num_head * self.head_dim)
        self.value = nn.Linear(hidden_dim, kv_num_head * self.head_dim)

        self.dropout = nn.Linear(dropout_rate)

        self.output = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, mask) -> torch.Tensor:
        bsz, seq_len, _ = X.shape

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        Q = Q.view(bsz, seq_len, self.q_num_head, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.kv_num_head, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.kv_num_head, self.head_dim).transpose(1, 2)

        K = K.repeat(1, self.num_per_group, 1, 1)
        V = V.repeat(1, self.num_per_group, 1, 1)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = torch.matmul(attn_scores, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.output(attn_output)

        return attn_output


class SafeSoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(X, dim=-1, keep_dim=True)
        exp_X = torch.exp(X - max_val)
        return exp_X / torch.sum(exp_X, dim=-1, keep_dim=True)


class SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, 4*dim)
        self.v = nn.Linear(dim, 4*dim)
        self.act = nn.SiLU()
        self.o = nn.Linear(4*dim, dim)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.o(self.act(self.w(X) * self.v(X)))


class LayerNorm(nn.Module):
    def __init__(self, eps: float, hidden_dim: int) -> None:
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_mean = torch.mean(X, dim=-1, keep_dim=True)
        X_var = torch.var(X, dim=-1, keep_dim=True)
        
        X_norm = (X - X_mean) / torch.sqrt(X_var + self.eps)
        return X_norm * self.gamma + self.beta


class RMSNorm(nn.Module):
    def __init__(self, eps: float, hidden_dim: int) -> None:
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(torch.pow(X, 2), dim=-1, keep_dim=True))
        X_norm = X / (rms + self.eps)
        return self.gamma * X_norm