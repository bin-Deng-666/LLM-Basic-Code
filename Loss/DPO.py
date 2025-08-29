import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, policy_chosen_logprobs, policy_rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs):
        log_chosen_ratio = policy_chosen_logprobs - ref_chosen_logprobs
        log_rejected_ratio = policy_rejected_logprobs - ref_rejected_logprobs
        
        diff = log_chosen_ratio - log_rejected_ratio
        reward_diff = self.beta * diff
        dpo_loss = -F.logsigmoid(reward_diff).mean()
        return dpo_loss
