import torch
from models.hamiltonian.utils import hamiltonian_vector_field

def hamiltonian_residual(trajectories, time_points, hamiltonian_fn):
    """
    Compute residual for a Hamiltonian system
    
    Args:
        trajectories: Trajectory data [batch_size, time_points]
        time_points: Time points [time_points, 1]
        hamiltonian_fn: Hamiltonian function H(z) where z is concatenated [q, p]
        
    Returns:
        residual: Residual of the Hamiltonian system
    """
    batch_size, num_points = trajectories.shape
    
    # Compute time step
    dt = time_points[1] - time_points[0] if num_points > 1 else torch.tensor(1.0, device=trajectories.device)
    
    # Compute derivatives using finite differences
    dX_dt = torch.zeros_like(trajectories)
    d2X_dt2 = torch.zeros_like(trajectories)
    
    # First derivatives (velocity)
    if num_points > 2:
        # Forward difference for first point
        dX_dt[:, 0] = (trajectories[:, 1] - trajectories[:, 0]) / dt
        
        # Central difference for interior points
        for i in range(1, num_points-1):
            dX_dt[:, i] = (trajectories[:, i+1] - trajectories[:, i-1]) / (2 * dt)
        
        # Backward difference for last point
        dX_dt[:, -1] = (trajectories[:, -1] - trajectories[:, -2]) / dt
        
        # Second derivatives (acceleration)
        # Forward difference for first point
        d2X_dt2[:, 0] = (dX_dt[:, 1] - dX_dt[:, 0]) / dt
        
        # Central difference for interior points
        for i in range(1, num_points-1):
            d2X_dt2[:, i] = (dX_dt[:, i+1] - dX_dt[:, i-1]) / (2 * dt)
        
        # Backward difference for last point
        d2X_dt2[:, -1] = (dX_dt[:, -1] - dX_dt[:, -2]) / dt
    
    # We need to compute the Hamiltonian residual point by point over the time dimension
    hamiltonian_residual = torch.zeros_like(trajectories)
    
    # Create a wrapper for the hamiltonian function that works with 1D inputs
    def hamiltonian_wrapper(z_1d):
        # Reshape to match expected input format [batch_size, 2*latent_dim]
        # For a 1D Hamiltonian system, latent_dim = 1
        return hamiltonian_fn(z_1d)
    
    # For a 1D Hamiltonian system, we treat position as the trajectory
    # and momentum as the velocity (dX_dt)
    for t in range(num_points):
        # Extract position and momentum at time t for all batch elements
        q_t = trajectories[:, t:t+1]  # [batch_size, 1]
        p_t = dX_dt[:, t:t+1]         # [batch_size, 1]
        
        # Combine to create input to Hamiltonian
        z_t = torch.cat([q_t, p_t], dim=1)  # [batch_size, 2]
        
        # Try to compute the Hamiltonian vector field
        try:
            # Compute dx_dt and dp_dt from Hamilton's equations
            _, dp_dt = hamiltonian_vector_field(hamiltonian_wrapper, q_t, p_t)
            
            # Residual: actual acceleration minus expected acceleration from Hamilton's equations
            hamiltonian_residual[:, t] = d2X_dt2[:, t] - dp_dt.squeeze()
        except Exception as e:
            # If there's an error, set residual to zero and continue
            print(f"Warning: Error computing Hamiltonian residual at time {t}: {e}")
            hamiltonian_residual[:, t] = torch.zeros(batch_size, device=trajectories.device)
    
    return hamiltonian_residual