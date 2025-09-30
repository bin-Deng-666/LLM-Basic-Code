import torch.nn as nn
import torch

class BatchNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float, momentum: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(num_feathidden_dimures))
        
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用缩放和偏移
        output = self.gamma * x_normalized + self.beta
        return output