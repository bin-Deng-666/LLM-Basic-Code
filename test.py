import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, logits, labels):
        exp_logits = torch.exp(logits)
        softmax_logits = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        
        true_class_logits = softmax_logits[torch.range(logits.shape[0]), labels]
        log_probs = -torch.log(true_class_logits)
        loss = torch.mean(log_probs)
        return loss