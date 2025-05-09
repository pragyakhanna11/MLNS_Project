import torch
import torch.nn as nn
from ..base.encoder import Encoder
from ..normalizing_flow.flow_decoder import NormalizingFlowDecoder

class PhysicsInformedNF(nn.Module):
    """
    Physics-Informed Variational Autoencoder with Normalizing Flow decoder
    
    This model extends the base PIVAE by using a normalizing flow decoder
    for improved uncertainty quantification.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, output_dim=32, 
                flow_type='realnvp', num_flows=3):
        super(PhysicsInformedNF, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder (standard VAE encoder)
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=3
        )
        
        # Normalizing Flow decoder
        self.decoder = NormalizingFlowDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            flow_type=flow_type,
            num_flows=num_flows
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
        return self.decoder(z, x)
    
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
            log_det: Log determinant of Jacobian for the flow
        """
        mu, s = self.encode(trajectory)
        z = self.reparameterize(mu, s)
        reconstructed, log_det = self.decoder(z, time_points, return_log_det=True)
        return reconstructed, mu, s, log_det
    
    def sample(self, num_samples, time_points):
        """
        Generate samples from the model
        
        Args:
            num_samples: Number of samples to generate
            time_points: Time points [time_points, 1]
            
        Returns:
            samples: Generated samples [num_samples, time_points]
        """
        z = torch.randn(num_samples, self.latent_dim, device=time_points.device)
        return self.decoder.sample(num_samples, time_points, z)
