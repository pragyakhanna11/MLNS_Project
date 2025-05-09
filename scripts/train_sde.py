#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the Physics-Informed VAE with Neural SDE encoder
on cell migration data.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import modules
from data.loader import load_cell_migration_data
from models.composite.pi_sde import PhysicsInformedNeuralSDE
from training.trainer import Trainer

def main():
    """Main function to train the Neural SDE-based PIVAE model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed VAE with Neural SDE encoder"
    )
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to processed cell migration data")
    parser.add_argument("--output-dir", "-o", type=str, default="results/sde_model",
                        help="Directory to save model and results")
    parser.add_argument("--window-size", "-w", type=int, default=20,
                        help="Window size for trajectory segments")
    parser.add_argument("--latent-dim", "-l", type=int, default=8,
                        help="Dimension of latent space")
    parser.add_argument("--hidden-dim", "-d", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--output-dim", type=int, default=32,
                        help="Output dimension of modal and basis networks")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", "-e", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--use-physics", action="store_true", default=True,
                        help="Use physics constraints in loss")
    parser.add_argument("--no-physics", dest="use_physics", action="store_false",
                        help="Disable physics constraints in loss")
    parser.add_argument("--progressive-physics", action="store_true", default=True,
                        help="Progressively increase physics weight")
    parser.add_argument("--physics-model", type=str, default="neural_sde",
                        choices=["ou", "persistence", "neural_sde"],
                        help="Physics model to use")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if args.input.endswith('.npy'):
        # Load preprocessed numpy array
        cell_trajectories = np.load(args.input)
    else:
        # Load and preprocess raw data
        cell_trajectories, _ = load_cell_migration_data(args.input, normalize=True)
    
    print(f"Loaded cell trajectories with shape: {cell_trajectories.shape}")
    
    # Create model
    model = PhysicsInformedNeuralSDE(
        input_dim=args.window_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.999, 0.999))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data=cell_trajectories,
        device=device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_type='sde'
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    losses, loss_history = trainer.train(
        num_epochs=args.epochs,
        use_physics=args.use_physics,
        progressive_physics=args.progressive_physics,
        physics_model=args.physics_model
    )
    
    # Generate samples
    print("Generating samples from trained model...")
    samples = trainer.generate_samples(num_samples=10, noise_scale=0.05)
    
    # Visualize SDE paths
    print("Visualizing SDE paths in latent space...")
    visualize_sde_paths(model, cell_trajectories, trainer.time_tensor, args.output_dir)
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write("Physics-Informed VAE with Neural SDE Training Configuration\n")
        f.write("=====================================================\n\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nFinal loss: {losses[-1]:.6f}\n")
        f.write(f"Final KL loss: {loss_history['kl'][-1]:.6f}\n")
        f.write(f"Final reconstruction loss: {loss_history['recon'][-1]:.6f}\n")
        f.write(f"Final physics loss: {loss_history['physics'][-1]:.6f}\n")
    
    print(f"Training complete! Results saved to {args.output_dir}")

def visualize_sde_paths(model, trajectories, time_points, output_dir):
    """
    Visualize the SDE paths in latent space
    
    Args:
        model: Trained model
        trajectories: Cell trajectory data
        time_points: Time points
        output_dir: Output directory
    """
    # Set model to evaluation mode
    model.eval()
    
    # Select a few examples
    batch_size = min(5, len(trajectories))
    examples = torch.tensor(trajectories[:batch_size], dtype=torch.float32).to(time_points.device)
    
    # Get SDE paths
    with torch.no_grad():
        _, _, z_paths = model.encode(examples, time_points)
    
    # Convert to numpy for plotting
    z_paths_np = z_paths.cpu().numpy()
    time_points_np = time_points.squeeze().cpu().numpy()
    
    # Plot latent trajectories
    plt.figure(figsize=(12, 8))
    
    # If latent dimension > 2, plot the first 2 dimensions
    for i in range(batch_size):
        # Get path for this example
        path = z_paths_np[:, i, :]
        
        # Plot first two latent dimensions
        plt.plot(path[:, 0], path[:, 1], '-o', markersize=3, label=f'Example {i+1}')
        # Mark start and end
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=8)  # Start (green)
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8)  # End (red)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Neural SDE Paths in Latent Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "sde_latent_paths.png"))
    plt.close()
    
    # Plot drift and diffusion for a single example
    example = examples[0:1]  # Take first example
    
    # Create a grid in latent space
    x1 = np.linspace(-3, 3, 20)
    x2 = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate drift and diffusion at each grid point
    drift_field = np.zeros((len(x1), len(x2), 2))
    diffusion_field = np.zeros((len(x1), len(x2)))
    
    with torch.no_grad():
        for i in range(len(x1)):
            for j in range(len(x2)):
                # Create latent state
                z = torch.tensor([[x1[i], x2[j]]], dtype=torch.float32).to(time_points.device)
                
                # Pad with zeros if latent_dim > 2
                if model.latent_dim > 2:
                    z = torch.cat([z, torch.zeros(1, model.latent_dim - 2, device=z.device)], dim=1)
                
                # Compute drift and diffusion
                t = time_points[0]  # Use first time point
                drift = model.encoder.drift(t, z).cpu().numpy()[0]
                diffusion = model.encoder.diffusion(t, z).cpu().numpy()[0]
                
                # Store first two dimensions of drift and mean diffusion
                drift_field[i, j, 0] = drift[0]
                drift_field[i, j, 1] = drift[1] if model.latent_dim > 1 else 0
                diffusion_field[i, j] = diffusion.mean()
    
    # Plot vector field of drift
    plt.figure(figsize=(10, 8))
    plt.quiver(X1, X2, drift_field[:, :, 0], drift_field[:, :, 1], 
              alpha=0.8, scale=20)
    
    # Overlay SDE paths
    for i in range(batch_size):
        path = z_paths_np[:, i, :]
        if path.shape[1] >= 2:
            plt.plot(path[:, 0], path[:, 1], '-', alpha=0.6, linewidth=1, label=f'Example {i+1}')
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Drift Vector Field in Latent Space')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "sde_drift_field.png"))
    plt.close()
    
    # Plot diffusion magnitude
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X1, X2, diffusion_field, shading='auto', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Diffusion Magnitude')
    
    # Overlay SDE paths
    for i in range(batch_size):
        path = z_paths_np[:, i, :]
        if path.shape[1] >= 2:
            plt.plot(path[:, 0], path[:, 1], 'w-', alpha=0.8, linewidth=1)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Diffusion Magnitude in Latent Space')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "sde_diffusion_field.png"))
    plt.close()
    
    # Plot a few trajectory samples with uncertainty
    plt.figure(figsize=(12, 8))
    
    # Generate multiple samples for the first example
    with torch.no_grad():
        # Get distribution parameters
        mu, s, _ = model.encode(example, time_points)
        
        # Generate multiple samples
        num_samples = 20
        samples = []
        
        for _ in range(num_samples):
            # Sample from latent distribution
            eps = torch.randn_like(s)
            z0 = mu + eps * s
            
            # Integrate SDE with different noise
            z_path = model.encoder.sdeint(z0, time_points)
            
            # Decode
            sample = model.decode(z_path[-1], time_points).squeeze().cpu().numpy()
            samples.append(sample)
    
    # Convert to array
    samples = np.array(samples)
    
    # Plot original trajectory
    time_np = time_points.squeeze().cpu().numpy()
    plt.plot(time_np, example.squeeze().cpu().numpy(), 'k-', linewidth=2, label='Original')
    
    # Plot samples
    for i in range(num_samples):
        plt.plot(time_np, samples[i], 'r-', alpha=0.2)
    
    # Plot mean and confidence interval
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    plt.plot(time_np, mean, 'r-', linewidth=2, label='Mean Reconstruction')
    plt.fill_between(time_np, mean - 2*std, mean + 2*std, color='r', alpha=0.2, label='±2σ')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Neural SDE: Trajectory Reconstruction with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "sde_reconstruction.png"))
    plt.close()
    
    # Analyze diffusion behavior over trajectory
    plt.figure(figsize=(12, 8))
    
    # Get the first example trajectory
    traj = example.squeeze().cpu().numpy()
    
    # Compute velocity using finite differences
    velocity = np.gradient(traj)
    
    # Get SDE path for this example
    z_path = z_paths_np[:, 0, :]
    
    # Compute diffusion along the path
    diffusion_values = []
    time_points_tensor = time_points.squeeze()
    
    with torch.no_grad():
        for i in range(len(time_points_tensor)):
            t = time_points_tensor[i]
            z = torch.tensor(z_path[i:i+1], dtype=torch.float32).to(time_points.device)
            diffusion = model.encoder.diffusion(t, z).cpu().numpy()[0].mean()
            diffusion_values.append(diffusion)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot position
    ax1.plot(time_np, traj, 'b-', linewidth=2)
    ax1.set_ylabel('Position')
    ax1.set_title('Cell Position')
    ax1.grid(True, alpha=0.3)
    
    # Plot velocity
    ax2.plot(time_np, velocity, 'g-', linewidth=2)
    ax2.set_ylabel('Velocity')
    ax2.set_title('Cell Velocity')
    ax2.grid(True, alpha=0.3)
    
    # Plot diffusion
    ax3.plot(time_np, diffusion_values, 'r-', linewidth=2)
    ax3.set_ylabel('Diffusion')
    ax3.set_xlabel('Time')
    ax3.set_title('Diffusion Coefficient')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sde_diffusion_analysis.png"))
    plt.close()

if __name__ == "__main__":
    main()