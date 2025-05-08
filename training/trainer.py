import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .loss import compute_loss

class Trainer:
    """
    Trainer for Physics-Informed VAE models
    """
    def __init__(self, model, optimizer, data, device='cpu', batch_size=32, 
                output_dir='results', model_type='base'):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model_type = model_type
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for training"""
        from data.loader import create_training_data
        
        # Create training data with sliding windows
        X, Y, time_indices = create_training_data(self.data)
        
        # Create normalized time points
        window_size = X.shape[1]
        time_points = np.linspace(0, 1, window_size)
        
        # Convert to tensors
        self.X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self.device)
        self.time_tensor = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        print(f"Prepared {len(X)} training samples with window size {window_size}")
    
    def train(self, num_epochs, use_physics=True, progressive_physics=True):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            use_physics: Whether to use physics constraints
            progressive_physics: Whether to progressively increase physics weight
        """
        # Training parameters
        sigma_obs = 1e-2
        sigma_phys_init = 100.0
        sigma_phys_final = 1.0
        
        # Initialize tracking variables
        losses = []
        best_loss = float('inf')
        best_state = None
        start_time = time.time()
        
        # Create progress bar
        progress_bar = tqdm(range(num_epochs), desc="Training")
        
        # Training loop
        for epoch in progress_bar:
            epoch_loss = 0
            batches = 0
            
            # Progressive increase of physics importance
            if progressive_physics:
                # Start with very little physics influence and gradually increase
                progress = min(1.0, (epoch / num_epochs) ** 2)  # Squared for slower ramp-up
                sigma_phys = sigma_phys_init * (1 - progress) + sigma_phys_final * progress
                # Only start adding physics after 20% of training
                physics_weight = 0.0 if epoch < num_epochs * 0.2 else min(1.0, (epoch - num_epochs * 0.2) / (num_epochs * 0.5))
            else:
                sigma_phys = sigma_phys_final
                physics_weight = 1.0 if use_physics else 0.0
            
            # Process data in batches
            indices = torch.randperm(len(self.X_tensor)).to(self.device)
            
            for i in range(0, len(self.X_tensor), self.batch_size):
                # Get batch indices
                batch_idx = indices[i:i+self.batch_size]
                if len(batch_idx) < 2:  # Skip too small batches
                    continue
                    
                x_batch = self.X_tensor[batch_idx]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute loss
                loss, loss_components = compute_loss(
                    self.model, x_batch, self.time_tensor,
                    sigma_obs=sigma_obs, sigma_phys=sigma_phys,
                    use_physics=use_physics, physics_weight=physics_weight
                )
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {i//self.batch_size}")
                    continue
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Track loss
                epoch_loss += loss_components['total']
                batches += 1
            
            # Skip reporting if no valid batches
            if batches == 0:
                continue
                
            # Average loss for epoch
            avg_loss = epoch_loss / batches
            losses.append(avg_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'physics_weight': f"{physics_weight:.2f}"
            })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.model.state_dict().copy()
                torch.save(best_state, os.path.join(self.output_dir, f"{self.model_type}_best_model.pt"))
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time/60:.2f} minutes")
        print(f"Best loss: {best_loss:.4f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Plot loss history
        self.plot_loss_history(losses)
        
        return losses
    
    def plot_loss_history(self, losses):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss - {self.model_type.upper()} Model')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_loss.png"))
        plt.close()
    
    def generate_samples(self, num_samples=10, time_points=None, noise_scale=0.1):
        """
        Generate trajectory samples from the trained model
        
        Args:
            num_samples: Number of samples to generate
            time_points: Optional custom time points for generation
        """
        # Use default time points if not provided
        if time_points is None:
            time_points = self.time_tensor
        
        # Set model to evaluation mode
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample from prior
                z = torch.randn(1, self.model.latent_dim, device=self.device)
                
                # Generate sample
                sample = self.model.decode(z, time_points).squeeze()
            
                # Add small noise to prevent perfectly straight lines
                noise = torch.randn_like(sample) * noise_scale
                sample = (sample + noise).cpu().numpy()
                samples.append(sample)
        
        # Convert to array
        samples = np.array(samples)
        
        # Plot samples
        plt.figure(figsize=(12, 8))
        time_values = time_points.squeeze().cpu().numpy()
        
        for i in range(num_samples):
            plt.plot(time_values, samples[i], alpha=0.7, linewidth=2)
        
        plt.xlabel('Time (normalized)', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.title(f'Generated Trajectories - {self.model_type.upper()} Model', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_samples.png"))
        plt.close()
        
        # Plot mean and uncertainty
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.plot(time_values, mean, 'k-', linewidth=2, label='Mean')
        plt.fill_between(time_values, mean - 2*std, mean + 2*std, alpha=0.3, color='gray', label='±2σ')
        plt.xlabel('Time (normalized)', fontsize=12)
        plt.ylabel('Position', fontsize=12)
        plt.title(f'Mean and Uncertainty - {self.model_type.upper()} Model', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, f"{self.model_type}_uncertainty.png"))
        plt.close()
        
        return samples