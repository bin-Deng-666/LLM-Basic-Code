import torch

def compute_mse_gradient(y, w, x, b):
    n = y.shape[0]
    y_hat = w * x + b 
    loss = torch.sum((y - y_hat) ** 2) / n 
    dw = -2 * torch.sum((y - y_hat) * x) / n
    db = -2 * torch.sum(y - y_hat) / n

    return loss, dw, db