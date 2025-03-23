import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, logits, labels):
        # 计算softmax
        exp_logits = torch.exp(logits)
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
        
        # 获得每个样本的真实类别概率
        batch_size = logits.shape[0]
        true_class_probs = softmax_probs[torch.arrange(batch_size), labels]
        
        # 计算对数概率
        log_probs = -torch.log(true_class_probs)
        
        # 计算平均损失
        loss = torch.mean(log_probs)
        
        return loss