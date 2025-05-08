import torch

def ou_process_residual(trajectories, time_points, theta=0.1, mu=0.0, sigma=0.5):
    """
    Compute residual for Ornstein-Uhlenbeck process
    dX = θ(μ - X)dt + σdW
    
    Args:
        trajectories: Trajectory data [batch_size, time_points]
        time_points: Time points [time_points, 1]
        theta: Mean reversion rate
        mu: Long-term mean
        sigma: Diffusion coefficient
        
    Returns:
        residual: Residual of the SDE [batch_size, time_points]
    """
    batch_size, num_points = trajectories.shape
    
    # Compute time step
    dt = time_points[1] - time_points[0] if num_points > 1 else torch.tensor(1.0)
    
    # Compute derivatives using finite differences
    dX_dt = torch.zeros_like(trajectories)
    
    # Forward difference for first point
    if num_points > 1:
        dX_dt[:, 0] = (trajectories[:, 1] - trajectories[:, 0]) / dt
        
        # Central difference for interior points
        for i in range(1, num_points-1):
            dX_dt[:, i] = (trajectories[:, i+1] - trajectories[:, i-1]) / (2 * dt)
        
        # Backward difference for last point
        dX_dt[:, -1] = (trajectories[:, -1] - trajectories[:, -2]) / dt
    
    # Compute drift term: θ(μ - X)
    drift = theta * (mu - trajectories)
    
    # Compute residual: dX/dt - θ(μ - X)
    residual = dX_dt - drift
    
    return residual