import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int=16,
                 lora_alpha: int=16):
        super().__init__()
        
        # Original linaer layer
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        # Two lower rank matrixs
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 1 / math.sqrt(rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

        # Get the scale parameter
        self.scale = lora_alpha / rank
    
    def forward(self, x):
        # Get the combined weights
        weight = self.linear.weight + self.scale * (self.lora_a @ self.lora_b).transpose(0, 1)

        # Go through the linear layer
        output = F.linear(x, weight, self.linear.bias)

        return output
        
if __name__ == "__main__":

    # Example usage of LoRA module
    model = LoRA(in_features=32, out_features=64)

    # Create random input tensor
    x = torch.randn(10, 32)  # Batch size 10, input features 32

    # Forward pass
    y = model(x)

    print("Output shape:", y.shape)
    
    # Check gradients
    loss = y.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} grad:\n{param.grad}")