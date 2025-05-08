import torch
import torch.nn as nn
from .resnet import ResNet

class ModalNet(nn.Module):
    """
    Modal network to learn spatial features
    Maps spatial coordinates to spatial features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(ModalNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks)
        
    def forward(self, x):
        return self.net(x)

class BasisNet(nn.Module):
    """
    Basis network to learn stochastic features
    Maps latent variables to stochastic features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(BasisNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks)
        
    def forward(self, z):
        return self.net(z)