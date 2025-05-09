import torch
import torch.nn as nn
import torch.nn.functional as F

def hamiltonian_vector_field(hamiltonian, x, p):
    """
    Compute the Hamiltonian vector field for the Hamiltonian H(x, p)
    
    Args:
        hamiltonian: Hamiltonian function H(x, p)
        x: Position variables [batch_size, dim]
        p: Momentum variables [batch_size, dim]
        
    Returns:
        dx_dt: Time derivative of x [batch_size, dim]
        dp_dt: Time derivative of p [batch_size, dim]
    """
    # Concatenate position and momentum for computing gradients
    z = torch.cat([x, p], dim=1)
    z.requires_grad_(True)
    
    # Compute Hamiltonian
    h = hamiltonian(z)
    
    # Compute gradients
    dh_dz = torch.autograd.grad(h.sum(), z, create_graph=True)[0]
    
    # Extract gradients with respect to position and momentum
    dh_dx = dh_dz[:, :x.shape[1]]
    dh_dp = dh_dz[:, x.shape[1]:]
    
    # Hamilton's equations
    dx_dt = dh_dp        # dx/dt = ∂H/∂p
    dp_dt = -dh_dx       # dp/dt = -∂H/∂x
    
    return dx_dt, dp_dt

def symplectic_integrator(hamiltonian, x0, p0, dt, steps=1):
    """
    Symplectic integrator for Hamiltonian systems
    
    Args:
        hamiltonian: Hamiltonian function H(x, p)
        x0: Initial position [batch_size, dim]
        p0: Initial momentum [batch_size, dim]
        dt: Time step
        steps: Number of integration steps
        
    Returns:
        x: Updated position [batch_size, dim]
        p: Updated momentum [batch_size, dim]
    """
    x, p = x0, p0
    
    for _ in range(steps):
        # Half step in momentum
        dx_dt, dp_dt = hamiltonian_vector_field(hamiltonian, x, p)
        p_half = p + 0.5 * dt * dp_dt
        
        # Full step in position
        dx_dt, dp_dt = hamiltonian_vector_field(hamiltonian, x, p_half)
        x_next = x + dt * dx_dt
        
        # Half step in momentum again
        dx_dt, dp_dt = hamiltonian_vector_field(hamiltonian, x_next, p_half)
        p_next = p_half + 0.5 * dt * dp_dt
        
        x, p = x_next, p_next
    
    return x, p