import torch

def normalizing_flow_residual(trajectories, time_points, flow_decoder, prior_variance=1.0):
    """
    Compute residual for normalizing flow model
    
    Args:
        trajectories: Trajectory data [batch_size, time_points]
        time_points: Time points [time_points, 1]
        flow_decoder: Normalizing flow decoder
        prior_variance: Variance of the prior distribution
        
    Returns:
        residual: Residual of the model
    """
    batch_size, num_points = trajectories.shape
    
    # Sample from standard normal prior
    z = torch.randn(batch_size, flow_decoder.latent_dim, device=trajectories.device)
    
    # Generate trajectories
    generated, log_det = flow_decoder(z, time_points, return_log_det=True)
    
    # Compute derivatives using finite differences
    dX_dt = torch.zeros_like(trajectories)
    
    if num_points > 2:
        # Forward difference for first point
        dX_dt[:, 0] = (trajectories[:, 1] - trajectories[:, 0]) / (time_points[1] - time_points[0])
        
        # Central difference for interior points
        for i in range(1, num_points-1):
            dt = (time_points[i+1] - time_points[i-1]) / 2
            dX_dt[:, i] = (trajectories[:, i+1] - trajectories[:, i-1]) / dt
        
        # Backward difference for last point
        dX_dt[:, -1] = (trajectories[:, -1] - trajectories[:, -2]) / (time_points[-1] - time_points[-2])
    
    # Compute residual as difference between real and generated trajectories
    trajectory_residual = trajectories - generated
    
    # Compute log-likelihood under flow model
    # log p(x) = log p(z) + log |det(dz/dx)|
    log_prior = -0.5 * prior_variance * torch.sum(z**2, dim=1) - 0.5 * flow_decoder.latent_dim * torch.log(torch.tensor(2 * torch.pi * prior_variance))
    log_likelihood = log_prior + log_det
    
    # Combine residuals (weighted sum)
    likelihood_weight = 0.1
    residual = trajectory_residual - likelihood_weight * log_likelihood.unsqueeze(1)
    
    return residual
