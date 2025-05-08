import torch
import torch.nn as nn
from ..base.encoder import Encoder
from ..base.decoder import BasisNet, ModalNet

class PhysicsInformedVAE(nn.Module):
    """
    Physics-Informed Variational Autoencoder for cell migration data
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, output_dim=32):
        super(PhysicsInformedVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=3
        )
        
        # Decoder components
        self.basis_net = BasisNet(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=3
        )
        
        self.modal_net = ModalNet(
            input_dim=1,  # Time dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=3
        )
    
    def encode(self, x):
        """Encode observations to latent space"""
        return self.encoder(x)
    
    def reparameterize(self, mu, s):
        """Reparameterization trick for sampling"""
        eps = torch.randn_like(s)
        return mu + eps * s
    
    def decode(self, z, x):
        """
        Decode from latent space to trajectory
        
        Args:
            z: Latent variables [batch_size, latent_dim]
            x: Time points [time_points, 1]
            
        Returns:
            Decoded trajectories [batch_size, time_points]
        """
        # Reshape inputs if needed
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # Add batch dimension
        
        # Get outputs from basis and modal networks
        psi = self.basis_net(z)  # [batch_size, output_dim]
        c = self.modal_net(x)    # [time_points, output_dim]
        
        # Compute inner product
        trajectory = torch.matmul(psi, c.transpose(0, 1))  # [batch_size, time_points]
        
        return trajectory
    
    def forward(self, trajectory, x):
        """
        Forward pass through the model
        
        Args:
            trajectory: Input trajectory [batch_size, time_points]
            x: Time points [time_points, 1]
            
        Returns:
            reconstructed: Reconstructed trajectory
            mu: Mean of latent distribution
            s: Standard deviation of latent distribution
            z: Sampled latent variables
        """
        mu, s = self.encode(trajectory)
        z = self.reparameterize(mu, s)
        reconstructed = self.decode(z, x)
        return reconstructed, mu, s, z