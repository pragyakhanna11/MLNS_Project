import torch
from physics.ou_process import ou_process_residual

def kl_divergence(mu, s):
    """
    Compute KL divergence between N(mu, s^2) and N(0, 1)
    
    Args:
        mu: Mean of approximate posterior [batch_size, latent_dim]
        s: Standard deviation of approximate posterior [batch_size, latent_dim]
        
    Returns:
        KL divergence for each sample in batch
    """
    return -0.5 * torch.sum(1 + 2*torch.log(s) - mu.pow(2) - s.pow(2), dim=1)

def compute_loss(model, x_batch, time_points, sigma_obs=1e-2, sigma_phys=0.1, 
                use_physics=True, physics_weight=5.0):
    """
    Compute ELBO loss with physics constraints
    
    Args:
        model: Physics-informed VAE model
        x_batch: Input trajectory batch [batch_size, time_points]
        time_points: Time points tensor [time_points, 1]
        sigma_obs: Observation noise
        sigma_phys: Physics constraint weight
        use_physics: Whether to include physics constraints
        physics_weight: Weight for physics term
        
    Returns:
        loss: Total loss
        loss_components: Dictionary of loss components
    """
    # Encode trajectories
    mu, s = model.encode(x_batch)
    
    # KL divergence
    kl_div = kl_divergence(mu, s)
    
    # Sample from latent space
    z = model.reparameterize(mu, s)
    
    # Reconstruct trajectories
    reconstructed = model.decode(z, time_points)
    
    # Reconstruction loss
    recon_loss = torch.mean((x_batch - reconstructed)**2) / (2 * sigma_obs**2)
    
    # Initialize physics loss
    physics_loss = torch.tensor(0.0, device=x_batch.device)
    
    # Add physics-informed loss if enabled
    if use_physics:
        # Compute residual for Ornstein-Uhlenbeck process
        residual = ou_process_residual(reconstructed, time_points)
        physics_loss = torch.mean(residual**2) / (2 * sigma_phys**2) * physics_weight
    
    # Total loss
    total_loss = torch.mean(kl_div) + recon_loss + physics_loss
    
    # Return loss components for logging
    loss_components = {
        'total': total_loss.item(),
        'kl': torch.mean(kl_div).item(),
        'recon': recon_loss.item(),
        'physics': physics_loss.item() if use_physics else 0.0
    }
    
    return total_loss, loss_components