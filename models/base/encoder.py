import torch
import torch.nn as nn
from .resnet import ResNet

class Encoder(nn.Module):
    """
    Encoder network that maps observations to latent space parameters
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks=3):
        super(Encoder, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, 2*latent_dim, num_blocks)
        self.latent_dim = latent_dim
        
    def forward(self, x):
        params = self.net(x)
        mu, log_var = params[:, :self.latent_dim], params[:, self.latent_dim:]
        
        # Ensure variance is positive
        s = torch.exp(0.5 * log_var)
        
        return mu, s