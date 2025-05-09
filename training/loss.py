import torch
from physics.ou_process import ou_process_residual
# from physics.persistence import persistent_random_walk_residual
from physics.hamiltonian import hamiltonian_residual
from physics.neural_sde import neural_sde_residual

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

def compute_loss(model, x_batch, time_points, sigma_obs=1e-2, sigma_phys=1.0, 
                use_physics=True, physics_weight=1.0, physics_model="persistence"):
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
        physics_model: Which physics model to use ("ou", "persistence", or "hamiltonian")
        
    Returns:
        loss: Total loss
        loss_components: Dictionary of loss components
    """
    # Check if model is HNN-based
    is_hnn_model = hasattr(model, 'get_hamiltonian_energy')
    
    # Encode trajectories
    if is_hnn_model:
        mu, s, energy = model.encoder(x_batch)
    else:
        mu, s = model.encode(x_batch)
        energy = None
    
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
    hamiltonian_loss = torch.tensor(0.0, device=x_batch.device)
    
    # Add physics-informed loss if enabled
    if use_physics:
        # Choose the appropriate physics model
        if physics_model == "ou":
            # Ornstein-Uhlenbeck process
            residual = ou_process_residual(reconstructed, time_points)
            physics_loss = torch.mean(residual**2) / (2 * sigma_phys**2) * physics_weight
        elif physics_model == "neural_sde" and hasattr(model.encoder, "drift") and hasattr(model.encoder, "diffusion"):
            # Get the drift and diffusion functions
            drift_fn = model.encoder.drift
            diffusion_fn = model.encoder.diffusion
            
            # Compute SDE residuals
            residual, diffusion_values = neural_sde_residual(
                reconstructed, time_points, drift_fn, diffusion_fn
            )
            
            # Physics loss based on SDE residuals
            physics_loss = torch.mean(residual**2) / (2 * sigma_phys**2) * physics_weight
            
            # Additional term to encourage appropriate diffusion levels
            # This helps prevent the model from setting diffusion to zero everywhere
            diffusion_reg = torch.mean((diffusion_values - 0.1)**2) * 0.01 * physics_weight
            physics_loss += diffusion_reg
        # elif physics_model == "persistence":
        #     # Persistent Random Walk
        #     residual = persistent_random_walk_residual(reconstructed, time_points)
        #     physics_loss = torch.mean(residual**2) / (2 * sigma_phys**2) * physics_weight
        elif physics_model == "hamiltonian" and is_hnn_model:
            # Hamiltonian system
            residual = hamiltonian_residual(
                reconstructed, time_points, 
                lambda z: model.encoder.hamiltonian(z)
            )
            physics_loss = torch.mean(residual**2) / (2 * sigma_phys**2) * physics_weight
            
            # Also add energy conservation loss
            hamiltonian_loss = torch.mean(energy**2) * physics_weight * 0.1
        else:
            raise ValueError(f"Unknown physics model: {physics_model}")
    
    # Total loss
    total_loss = torch.mean(kl_div) + recon_loss + physics_loss + hamiltonian_loss
    
    # Return loss components for logging
    loss_components = {
        'total': total_loss.item(),
        'kl': torch.mean(kl_div).item(),
        'recon': recon_loss.item(),
        'physics': physics_loss.item() if use_physics else 0.0,
        'hamiltonian': hamiltonian_loss.item() if is_hnn_model and use_physics else 0.0
    }
    
    return total_loss, loss_components