import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Used in the paper for all networks
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """
    Residual block with scaling factor to address gradient exploding problem
    The paper uses α = 0.3 for the scaling factor
    """
    def __init__(self, dim, scaling_factor=0.3):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = Swish()
        self.scaling_factor = scaling_factor
        
    def forward(self, x):
        return x + self.scaling_factor * self.activation(self.linear(x))