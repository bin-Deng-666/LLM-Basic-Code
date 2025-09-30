import torch
import torch.nn as nn

class GQA(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, group_num: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.group_num = group_num
        self.num_per_group = head_num // group_num
        
        self.query = nn.Linear(hidden_dim, self.head_num * self.head_dim)
        self.key = nn.Linear(hidden_dim, self.group_num * self.head_dim)
        self.value = nn.Linear(hidden_dim, self.group_num * self.head_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.output = nn.Linear(hidden_dim, hidden_dim)    
    
    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()
    
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)
        
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)

        K = K.repeat(1, self.num_per_group, 1, 1)
        V = V.repeat(1, self.num_per_group, 1, 1)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 调整掩码形状: [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf"))
        
        # 注意力归一化
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_scores, V)
        
        # 转换回[batch_size, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 应用输出投影
        output = self.output(attn_output)
        return output
        