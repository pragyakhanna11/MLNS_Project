import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.decoder import BasisNet, ModalNet
from .transforms import RealNVP, MAF, PlanarFlow

class NormalizingFlowDecoder(nn.Module):
    """
    Decoder based on normalizing flows for improved uncertainty quantification
    
    This decoder uses normalizing flows to transform a simple prior distribution
    to a more complex distribution, allowing for better uncertainty representation.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, flow_type='realnvp', num_flows=3):
        super(NormalizingFlowDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create flow layers
        self.flows = nn.ModuleList()
        
        for _ in range(num_flows):
            if flow_type == 'planar':
                flow = PlanarFlow(latent_dim)
            elif flow_type == 'realnvp':
                flow = RealNVP(latent_dim, hidden_dim)
            elif flow_type == 'maf':
                flow = MAF(latent_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
            
            self.flows.append(flow)
        
        # Basis and modal networks (similar to standard decoder)
        self.basis_net = BasisNet(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.modal_net = ModalNet(
            input_dim=1,  # Time dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    
    def forward(self, z, x, return_log_det=False):
        """
        Forward pass through the decoder
        
        Args:
            z: Latent variables [batch_size, latent_dim]
            x: Time points [time_points, 1]
            return_log_det: Whether to return log determinant of Jacobian
            
        Returns:
            trajectory: Decoded trajectory [batch_size, time_points]
            log_det: Log determinant of Jacobian (optional)
        """
        # Apply flows to transform the latent variable
        z_transformed = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        for flow in self.flows:
            z_transformed, log_det = flow(z_transformed)
            log_det_total += log_det
        
        # Apply basis and modal networks
        psi = self.basis_net(z_transformed)  # [batch_size, output_dim]
        c = self.modal_net(x)                # [time_points, output_dim]
        
        # Compute inner product
        trajectory = torch.matmul(psi, c.transpose(0, 1))  # [batch_size, time_points]
        
        if return_log_det:
            return trajectory, log_det_total
        else:
            return trajectory
    
    def sample(self, num_samples, x, z=None):
        """
        Generate samples from the decoder
        
        Args:
            num_samples: Number of samples to generate
            x: Time points [time_points, 1]
            z: Optional latent variables to use (otherwise sampled from prior)
            
        Returns:
            samples: Generated samples [num_samples, time_points]
        """
        device = next(self.parameters()).device
        
        # Sample from prior if z not provided
        if z is None:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.forward(z, x)
        
        return samples