import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.layers import Swish, ResidualBlock
from ..base.resnet import ResNet

class HamiltonianEncoder(nn.Module):
    """
    Hamiltonian Neural Network-based encoder for Physics-Informed VAE
    
    This encoder learns a latent space that respects Hamiltonian dynamics,
    which is particularly useful for cell migration data that exhibits
    oscillatory behaviors.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks=3):
        super(HamiltonianEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_net = ResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_blocks=num_blocks
        )
        
        # Hamiltonian network (energy function)
        self.h_net = nn.Sequential(
            nn.Linear(2*latent_dim, hidden_dim),  # Changed to take 2*latent_dim as input
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Position and momentum networks
        self.position_net = nn.Linear(hidden_dim, latent_dim)
        self.momentum_net = nn.Linear(hidden_dim, latent_dim)
        
        # Distribution parameters
        self.latent_net = nn.Linear(hidden_dim, 2*latent_dim)
        
    def hamiltonian(self, z):
        """
        Hamiltonian function
        
        Args:
            z: Combined position and momentum variables [batch_size, 2*latent_dim]
            
        Returns:
            H: Hamiltonian (energy) [batch_size, 1]
        """
        return self.h_net(z)
    
    def forward(self, x):
        """
        Forward pass through the encoder
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            s: Standard deviation of latent distribution [batch_size, latent_dim]
            energy: Hamiltonian energy of the system [batch_size, 1]
        """
        # Extract features
        features = self.feature_net(x)
        
        # Compute latent distribution parameters
        latent_params = self.latent_net(features)
        mu, log_var = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        
        # Standard deviation
        s = torch.exp(0.5 * log_var)
        
        # Compute position and momentum components
        q = self.position_net(features)  # Position-like variables [batch_size, latent_dim]
        p = self.momentum_net(features)  # Momentum-like variables [batch_size, latent_dim]
        
        # Compute Hamiltonian energy
        z = torch.cat([q, p], dim=1)  # [batch_size, 2*latent_dim]
        energy = self.hamiltonian(z)
        
        return mu, s, energy