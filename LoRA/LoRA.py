import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int=16,
                 lora_alpha: int=16,
                 dropout_rate: float=0.5):
        super().__init__()
        
        # Original linaer layer
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        # Two lower rank matrixs
        self.lora_a = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        # Get the scale parameter
        self.scale = lora_alpha / rank

        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Get the combined weights
        weight = self.linear.weight + self.scale * (self.lora_a @ self.lora_b).transpose(0, 1)

        # Go through the linear layer
        output = F.linear(x, weight, self.linear.bias)

        # Apply the dropout
        output = self.dropout(output)

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