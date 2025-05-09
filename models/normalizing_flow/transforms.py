import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base.layers import Swish

class PlanarFlow(nn.Module):
    """
    Planar flow transformation: f(z) = z + u * tanh(w^T * z + b)
    
    A simple normalizing flow that applies a planar transformation.
    """
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.dim = dim
        
        # Initialize parameters
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, z):
        """
        Transform z and compute log determinant of Jacobian
        
        Args:
            z: Input tensor [batch_size, dim]
            
        Returns:
            z_new: Transformed tensor [batch_size, dim]
            log_det: Log determinant of the Jacobian [batch_size]
        """
        # Ensure w^T u > -1 (needed for invertibility)
        wtu = (self.w @ self.u).unsqueeze(0)
        m_wtu = -1 + F.softplus(wtu)
        u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w)
        
        # Apply transformation
        z_new = z + u_hat.unsqueeze(0) * torch.tanh(z @ self.w + self.b)
        
        # Compute log determinant of Jacobian
        psi = (1 - torch.tanh(z @ self.w + self.b)**2) * self.w.unsqueeze(0)
        log_det = torch.log(torch.abs(1 + psi @ u_hat))
        
        return z_new, log_det

class RealNVP(nn.Module):
    """
    Real-valued Non-Volume Preserving (RealNVP) transformation
    
    A more powerful normalizing flow that splits dimensions and applies
    affine coupling layers.
    """
    def __init__(self, dim, hidden_dim=64, num_layers=3):
        super(RealNVP, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # For each layer, we need scale and translation networks
        # Alternate masking pattern for each layer
        self.masks = []
        self.scale_nets = nn.ModuleList()
        self.translation_nets = nn.ModuleList()
        
        for i in range(num_layers):
            # Alternate masking pattern (binary mask)
            if i % 2 == 0:
                mask = torch.zeros(dim)
                mask[:dim//2] = 1.0
            else:
                mask = torch.zeros(dim)
                mask[dim//2:] = 1.0
                
            self.masks.append(mask)
            
            # Scale and translation networks
            scale_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, dim),
                nn.Tanh()  # Using tanh for stability
            )
            
            translation_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, hidden_dim),
                Swish(),
                nn.Linear(hidden_dim, dim)
            )
            
            self.scale_nets.append(scale_net)
            self.translation_nets.append(translation_net)
    
    def forward(self, z):
        """
        Transform z and compute log determinant of Jacobian
        
        Args:
            z: Input tensor [batch_size, dim]
            
        Returns:
            z_new: Transformed tensor [batch_size, dim]
            log_det: Log determinant of the Jacobian [batch_size]
        """
        batch_size = z.shape[0]
        log_det_total = torch.zeros(batch_size, device=z.device)
        z_new = z
        
        for i in range(self.num_layers):
            mask = self.masks[i].to(z.device)
            
            # Split input based on mask
            z_masked = z_new * mask
            z_pass = z_new * (1 - mask)
            
            # Compute scale and translation
            scale = self.scale_nets[i](z_masked) * (1 - mask)
            translation = self.translation_nets[i](z_masked) * (1 - mask)
            
            # Apply transformation
            z_new = z_masked + z_pass * torch.exp(scale) + translation
            
            # Compute log determinant of Jacobian
            log_det = torch.sum(scale, dim=1)
            log_det_total += log_det
        
        return z_new, log_det_total
    
    def inverse(self, z):
        """
        Apply inverse transformation
        
        Args:
            z: Input tensor [batch_size, dim]
            
        Returns:
            z_orig: Inverse transformed tensor [batch_size, dim]
        """
        z_orig = z
        
        for i in range(self.num_layers - 1, -1, -1):
            mask = self.masks[i].to(z.device)
            
            # Split input based on mask
            z_masked = z_orig * mask
            
            # Compute scale and translation
            scale = self.scale_nets[i](z_masked) * (1 - mask)
            translation = self.translation_nets[i](z_masked) * (1 - mask)
            
            # Apply inverse transformation
            z_orig = z_masked + (z_orig - translation) * torch.exp(-scale) * (1 - mask)
        
        return z_orig

class MAF(nn.Module):
    """
    Masked Autoregressive Flow
    
    A powerful normalizing flow that uses autoregressive models to transform
    the latent space.
    """
    def __init__(self, dim, hidden_dim=64, num_layers=3):
        super(MAF, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Create autoregressive networks for each layer
        self.ar_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            # Each autoregressive network predicts scale and shift parameters
            ar_net = MADE(dim, hidden_dim, 2 * dim)
            self.ar_nets.append(ar_net)
    
    def forward(self, z):
        """
        Transform z and compute log determinant of Jacobian
        
        Args:
            z: Input tensor [batch_size, dim]
            
        Returns:
            z_new: Transformed tensor [batch_size, dim]
            log_det: Log determinant of the Jacobian [batch_size]
        """
        batch_size = z.shape[0]
        log_det_total = torch.zeros(batch_size, device=z.device)
        z_new = z
        
        for i in range(self.num_layers):
            # Get scale and shift parameters from autoregressive network
            params = self.ar_nets[i](z_new)
            scale = params[:, :self.dim]
            shift = params[:, self.dim:]
            
            # Apply transformation
            z_new = scale * z_new + shift
            
            # Compute log determinant of Jacobian
            log_det = torch.sum(scale, dim=1)
            log_det_total += log_det
        
        return z_new, log_det_total

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation
    
    A neural network with masked connections to enforce autoregressive property.
    Used as a building block for MAF.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create masks
        self.masks = self._create_masks()
        
        # Create masked layers
        self.net = nn.Sequential(
            MaskedLinear(input_dim, hidden_dim, self.masks[0]),
            Swish(),
            MaskedLinear(hidden_dim, hidden_dim, self.masks[1]),
            Swish(),
            MaskedLinear(hidden_dim, output_dim, self.masks[2])
        )
    
    def _create_masks(self):
        """Create masks for the autoregressive property"""
        # Assign degrees to input nodes (0 to input_dim-1)
        input_degrees = torch.arange(self.input_dim)
        
        # Assign random degrees to hidden nodes, ensuring they're in [0, input_dim-1]
        hidden_degrees = torch.randint(0, self.input_dim-1, (self.hidden_dim,))
        
        # For output, create output_dim//input_dim groups of degrees
        output_degrees = torch.arange(self.input_dim).repeat(self.output_dim // self.input_dim)
        
        # Mask for input to hidden
        mask_input_hidden = (input_degrees.unsqueeze(1) <= hidden_degrees.unsqueeze(0)).float()
        
        # Mask for hidden to hidden
        mask_hidden_hidden = (hidden_degrees.unsqueeze(1) <= hidden_degrees.unsqueeze(0)).float()
        
        # Mask for hidden to output
        mask_hidden_output = (hidden_degrees.unsqueeze(1) < output_degrees.unsqueeze(0)).float()
        
        return [mask_input_hidden, mask_hidden_hidden, mask_hidden_output]
    
    def forward(self, x):
        return self.net(x)

class MaskedLinear(nn.Module):
    """
    Linear layer with a mask applied to the weights
    
    Used in MADE to enforce autoregressive property.
    """
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)
