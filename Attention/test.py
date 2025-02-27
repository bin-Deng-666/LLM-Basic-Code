import torch
import math
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, head_num: int, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        self.query = nn.Linear(embed_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.value = nn.Linear(embed_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()
        
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_weights = attention_weights.masked_fill(
            attention_mask==0,
            float("-inf")
        )
        
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V).transpose(1, 2)
        output = output.view(batch_size, seq_len, self.hidden_dim)
        return output
        
        