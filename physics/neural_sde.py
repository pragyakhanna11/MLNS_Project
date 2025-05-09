import torch

def neural_sde_residual(trajectories, time_points, drift_fn, diffusion_fn, dt_scale=1.0):
    """
    Compute residuals for Neural SDE model
    
    Args:
        trajectories: Trajectory data [batch_size, time_points]
        time_points: Time points [time_points, 1]
        drift_fn: Drift function for the SDE
        diffusion_fn: Diffusion function for the SDE
        dt_scale: Scaling factor for time derivatives
        
    Returns:
        residual: Residual of the SDE [batch_size, time_points]
        diffusion_values: Diffusion values [batch_size, time_points]
    """
    batch_size, num_points = trajectories.shape
    
    # Compute time step
    dt = time_points[1] - time_points[0] if num_points > 1 else torch.tensor(1.0, device=trajectories.device)
    dt = dt * dt_scale
    
    # Compute derivatives using finite differences
    dX_dt = torch.zeros_like(trajectories)
    
    # First derivatives (velocity)
    if num_points > 2:
        # Forward difference for first point
        dX_dt[:, 0] = (trajectories[:, 1] - trajectories[:, 0]) / dt
        
        # Central difference for interior points
        for i in range(1, num_points-1):
            dX_dt[:, i] = (trajectories[:, i+1] - trajectories[:, i-1]) / (2 * dt)
        
        # Backward difference for last point
        dX_dt[:, -1] = (trajectories[:, -1] - trajectories[:, -2]) / dt
    
    # Initialize containers for drift and diffusion values
    drift_values = torch.zeros_like(trajectories)
    diffusion_values = torch.zeros_like(trajectories)
    
    # For each time point
    for i in range(num_points):
        t = time_points[i]
        x = trajectories[:, i:i+1]  # Current state [batch_size, 1]
        
        try:
            # Compute drift and diffusion
            drift = drift_fn(t, x)
            diffusion = diffusion_fn(t, x)
            
            # Store values
            drift_values[:, i] = drift.squeeze()
            diffusion_values[:, i] = diffusion.squeeze()
        except Exception as e:
            print(f"Error computing drift/diffusion at time {i}: {e}")
    
    # Residual: observed derivative minus predicted drift
    # Note: In the Stratonovich interpretation, the diffusion contributes to the mean dynamics
    residual = dX_dt - drift_values
    
    return residual, diffusion_values