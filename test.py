import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels):
        y_true_labels = logits[torch.arange(logits.shape[0]), labels]
        log_probs = -torch.log(y_true_labels)
        return torch.mean(log_probs)


def compute_mse_gradient(y, w, x, b):
    n = y.shape[0]
    y_hat = w * x + b 
    loss = torch.sum((y - y_hat) ** 2) / n 
    dw = -2 * torch.sum((y - y_hat) * x) / n
    db = -2 * torch.sum(y - y_hat) / n

    return loss, dw, db
