import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.layers import Swish, ResidualBlock
from ..base.resnet import ResNet
from .diffusion import euler_maruyama, DiffusionNet

class NeuralSDEEncoder(nn.Module):
    """
    Neural SDE-based encoder for Physics-Informed VAE
    
    This encoder replaces the standard VAE encoder with a latent space
    governed by a stochastic differential equation, allowing for
    time-dependent and state-dependent noise modeling.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks=3):
        super(NeuralSDEEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_net = ResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_blocks=num_blocks
        )
        
        # Initial state network
        self.initial_net = nn.Linear(hidden_dim, latent_dim)
        
        # Drift network (deterministic dynamics)
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Diffusion network (stochastic dynamics)
        self.diffusion_net = DiffusionNet(latent_dim, hidden_dim)
        
        # Distribution parameters (for variational inference)
        self.latent_net = nn.Linear(hidden_dim, 2*latent_dim)
    
    def drift(self, t, z):
        """
        Drift function for the SDE
        
        Args:
            t: Time point (scalar or batch)
            z: State [batch_size, latent_dim]
            
        Returns:
            drift: Drift term [batch_size, latent_dim]
        """
        # Ensure t has the right shape
        if isinstance(t, float) or t.dim() == 0:
            t_tensor = torch.full((z.shape[0], 1), t, device=z.device)
        else:
            t_tensor = t.view(-1, 1)
            
        # Concatenate time and state
        tz = torch.cat([t_tensor, z], dim=1)
        
        # Return drift
        return self.drift_net(tz)
    
    def diffusion(self, t, z):
        """
        Diffusion function for the SDE
        
        Args:
            t: Time point (scalar or batch)
            z: State [batch_size, latent_dim]
            
        Returns:
            diffusion: Diffusion term [batch_size, latent_dim]
        """
        return self.diffusion_net(t, z)
    
    def sdeint(self, z0, ts, method='euler', num_samples=1, adjoint=False):
        """
        Integrate the SDE
        
        Args:
            z0: Initial state [batch_size, latent_dim]
            ts: Time points [num_time_points]
            method: Integration method ('euler' or 'milstein')
            num_samples: Number of samples to generate
            adjoint: Whether to use adjoint method for backpropagation
            
        Returns:
            z_path: Path of the SDE [num_time_points, batch_size, latent_dim]
        """
        batch_size, latent_dim = z0.shape
        device = z0.device
        num_time_points = len(ts)
        
        # Initialize noise for all samples
        noise = torch.randn(num_samples, num_time_points-1, batch_size, latent_dim, device=device)
        
        # Initialize path storage
        z_paths = []
        
        # Generate multiple samples
        for i in range(num_samples):
            if method == 'euler':
                z_path = euler_maruyama(
                    drift_fn=self.drift,
                    diffusion_fn=self.diffusion,
                    z0=z0,
                    ts=ts,
                    noise=noise[i]
                )
            elif method == 'milstein':
                # For Milstein scheme, we would need the gradient of diffusion
                raise NotImplementedError("Milstein scheme not implemented yet")
            else:
                raise ValueError(f"Unknown SDE integration method: {method}")
            
            z_paths.append(z_path)
        
        # Return paths
        if num_samples == 1:
            return z_paths[0]
        else:
            return torch.stack(z_paths)
    
    def forward(self, x, integration_times=None):
        """
        Forward pass through the encoder
        
        Args:
            x: Input data [batch_size, input_dim]
            integration_times: Time points for SDE integration (optional)
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            s: Standard deviation of latent distribution [batch_size, latent_dim]
            z_path: Path of the SDE [num_time_points, batch_size, latent_dim]
        """
        # Extract features
        features = self.feature_net(x)
        
        # Compute latent distribution parameters
        latent_params = self.latent_net(features)
        mu, log_var = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        s = torch.exp(0.5 * log_var)
        
        # Compute initial state
        z0 = self.initial_net(features)
        
        # Define default integration times if not provided
        if integration_times is None:
            integration_times = torch.linspace(0, 1, 10, device=x.device)
        
        # Integrate the SDE
        z_path = self.sdeint(z0, integration_times)
        
        return mu, s, z_path
