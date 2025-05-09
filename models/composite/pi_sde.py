import torch
import torch.nn as nn
from ..base.decoder import BasisNet, ModalNet
from ..neural_sde.sde_encoder import NeuralSDEEncoder

class PhysicsInformedNeuralSDE(nn.Module):
    """
    Physics-Informed Variational Autoencoder with Neural SDE encoder
    
    This model extends the base PIVAE by using a Neural SDE encoder that
    captures time-dependent and state-dependent stochasticity, which is 
    particularly suitable for modeling cell migration with varying levels
    of randomness.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, output_dim=32):
        super(PhysicsInformedNeuralSDE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural SDE encoder
        self.encoder = NeuralSDEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=3
        )
        
        # Decoder components - same as base model
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
    
    def encode(self, x, integration_times=None):
        """Encode observations to latent space"""
        return self.encoder(x, integration_times)
    
    def get_sde_path(self, x, integration_times=None):
        """Get the full SDE path for the input"""
        _, _, z_path = self.encoder(x, integration_times)
        return z_path
    
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
    
    def forward(self, trajectory, time_points):
        """
        Forward pass through the model
        
        Args:
            trajectory: Input trajectory [batch_size, time_points]
            time_points: Time points [time_points, 1]
            
        Returns:
            reconstructed: Reconstructed trajectory
            mu: Mean of latent distribution
            s: Standard deviation of latent distribution
            z_path: Full SDE path in latent space
        """
        mu, s, z_path = self.encode(trajectory, time_points)
        
        # Use the final state from the SDE path for reconstruction
        z_final = z_path[-1]  
        
        reconstructed = self.decode(z_final, time_points)
        return reconstructed, mu, s, z_path