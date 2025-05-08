import torch
import torch.nn as nn
from .layers import Swish, ResidualBlock

class ResNet(nn.Module):
    """
    Residual network with multiple residual blocks
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, scaling_factor=0.3):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, scaling_factor) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = Swish()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)