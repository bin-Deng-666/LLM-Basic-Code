import torch.nn as nn
import torch
import math

class SelfAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 hidden_dim: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(embed_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.value = nn.Linear(embed_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X, attention_mask=None):
        # X's shape is (batch, seq, embed_dim)
        
        # Q K V's shape is (batch, seq, hidden_dim)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        # attention_weights's shape is (batch, seq, seq)
        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        
        # Apply the attention mask
        attention_weights = attention_weights.masked_fill(
            attention_mask == 0,
            float("-inf")
        )
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply softmax
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # output's shape is (batch, seq, hidden_dim)
        output = torch.matmul(attention_weights, V)
        
        return output
    
if __name__ == "__main__":
    embed_dim = 128
    hidden_dim = 256
    batch_size = 32
    seq_length = 10
    
    model = SelfAttention(embed_dim=embed_dim, hidden_dim=hidden_dim)
    X = torch.randn(batch_size, seq_length, embed_dim)
    
    # Create an example attention mask (all ones for simplicity)
    attention_mask = torch.ones(batch_size, seq_length, seq_length)
    
    output = model(X, attention_mask)
    print(output.shape)