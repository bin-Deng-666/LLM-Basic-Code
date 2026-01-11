import torch.nn as nn

class AdaNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gammar = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        mean = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True)
        x_normalized = (X - mean) / torch.sqrt(var)

        gammar = self.gammar(X)
        beta = self.beta(X)

        return gammar * x_normalized + beta

