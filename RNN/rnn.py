import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 记录需要的参数
        self.hidden_size = hidden_size
        # 隐藏状态需要的参数
        self.w_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.randn(hidden_size))
        # 输出需要的参数
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # 记录输入的形状大小
        batch_size, seq_len, _ = X.size()
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            x_i = X[:, i, :]
            # 更新隐藏状态
            h = torch.tanh(
                    torch.mm(x_i, self.w_xh) + 
                    torch.mm(h, self.w_hh) + 
                    self.bias_h.squeeze(1)
                )
        
        output = self.output(h)
        return output



