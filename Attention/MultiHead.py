import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                head_num: int,
                embed_dim: int,
                hidden_dim: int,
                dropout_rate: float=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        self.query = nn.Linear(embed_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.value = nn.Linear(embed_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X, attention_mask=None):
        # X's shape is (batch, seq, embed_dim)
        batch_size, seq_len, _ = X.size()
        
        # QKV's shape is (batch, seq, hidden_dim)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        # QKV's shape is (batch, head_num, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        # attention_weights's shape is (batch, head_num, seq, seq)
        attention_weights = torch.matmul(Q, V.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0,
                float('-inf')
            )
        
        # Apply the softmax
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Apply the dropout
        attention_weights = self.dropout(attention_weights)
        
        # output's shape is (batch, head_num, seq, head_dim)
        output = torch.matmul(attention_weights, V)
        
        # output's shape is (batch, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return output

if __name__ == "__main__":
    embed_dim = 128
    hidden_dim = 256
    head_num = 4
    batch_size = 32
    seq_length = 10

    model = MultiHeadSelfAttention(head_num=head_num, embed_dim=embed_dim, hidden_dim=hidden_dim)
    X = torch.randn(batch_size, seq_length, embed_dim)

    # Create an example attention mask (all ones for simplicity)
    attention_mask = torch.ones(batch_size, head_num, seq_length, seq_length)

    output = model(X, attention_mask)
    print(output.shape)