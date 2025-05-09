#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the Physics-Informed VAE with Normalizing Flow decoder
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
from models.composite.pi_nf import PhysicsInformedNF
from training.trainer import Trainer

def main():
    """Main function to train the Normalizing Flow-based PIVAE model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed VAE with Normalizing Flow decoder"
    )
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to processed cell migration data")
    parser.add_argument("--output-dir", "-o", type=str, default="results/nf_model",
                        help="Directory to save model and results")
    parser.add_argument("--window-size", "-w", type=int, default=20,
                        help="Window size for trajectory segments")
    parser.add_argument("--latent-dim", "-l", type=int, default=8,
                        help="Dimension of latent space")
    parser.add_argument("--hidden-dim", "-d", type=int, default=64,
                        help="Dimension of hidden layers")
    parser.add_argument("--output-dim", type=int, default=32,
                        help="Output dimension of modal and basis networks")
    parser.add_argument("--flow-type", type=str, default="realnvp",
                        choices=["planar", "realnvp", "maf"],
                        help="Type of normalizing flow to use")
    parser.add_argument("--num-flows", type=int, default=3,
                        help="Number of flow layers")
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
    parser.add_argument("--physics-model", type=str, default="normalizing_flow",
                        choices=["ou", "persistence", "normalizing_flow"],
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
    model = PhysicsInformedNF(
        input_dim=args.window_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        flow_type=args.flow_type,
        num_flows=args.num_flows
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
        model_type='nf'
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
    
    # Visualize flow transformations
    print("Visualizing flow transformations...")
    visualize_flow_transformations(model, cell_trajectories, trainer.time_tensor, args.output_dir)
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write("Physics-Informed VAE with Normalizing Flow Training Configuration\n")
        f.write("=========================================================\n\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nFinal loss: {losses[-1]:.6f}\n")
        f.write(f"Final KL loss: {loss_history['kl'][-1]:.6f}\n")
        f.write(f"Final reconstruction loss: {loss_history['recon'][-1]:.6f}\n")
        f.write(f"Final physics loss: {loss_history['physics'][-1]:.6f}\n")
    
    print(f"Training complete! Results saved to {args.output_dir}")

def visualize_flow_transformations(model, trajectories, time_points, output_dir):
    """
    Visualize how the normalizing flow transforms the latent space
    
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
    
    # Create grid in latent space
    if model.latent_dim >= 2:
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        # If latent dim > 2, pad with zeros
        if model.latent_dim > 2:
            pad = np.zeros((grid_points.shape[0], model.latent_dim - 2))
            grid_points = np.concatenate([grid_points, pad], axis=1)
        
        # Convert to tensor
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(time_points.device)
        
        # Transform grid through each flow
        transformed_grids = [grid_tensor.cpu().numpy()]
        
        with torch.no_grad():
            z = grid_tensor
            
            for flow in model.decoder.flows:
                z, _ = flow(z)
                transformed_grids.append(z.cpu().numpy())
        
        # Plot original and transformed grids
        for i, grid in enumerate(transformed_grids):
            plt.figure(figsize=(10, 8))
            
            # Take first two dimensions
            x = grid[:, 0].reshape(20, 20)
            y = grid[:, 1].reshape(20, 20)
            
            # Plot grid
            plt.scatter(x, y, c=np.sqrt(x**2 + y**2), cmap='viridis', s=5)
            
            # Add title
            if i == 0:
                plt.title('Original Latent Space (Standard Normal)')
            else:
                plt.title(f'After Flow Layer {i}')
            
            plt.xlabel('Latent Dim 1')
            plt.ylabel('Latent Dim 2')
            plt.colorbar(label='Distance from Origin')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"flow_transformation_{i}.png"))
            plt.close()
    
    # Visualize uncertainty in trajectory space
    with torch.no_grad():
        # Get latent representations
        mu, s = model.encode(examples)
        
        # Generate multiple samples for each example
        num_samples = 20
        
        for i in range(batch_size):
            plt.figure(figsize=(12, 8))
            
            # Original trajectory
            orig = examples[i].cpu().numpy()
            plt.plot(time_points.squeeze().cpu().numpy(), orig, 'k-', linewidth=2, label='Original')
            
            # Generate samples
            samples = []
            for _ in range(num_samples):
                # Sample latent
                z = mu[i:i+1] + s[i:i+1] * torch.randn_like(s[i:i+1])
                
                # Pass through flows and decode
                sample = model.decode(z, time_points)
                samples.append(sample.squeeze().cpu().numpy())
            
            # Plot samples
            for sample in samples:
                plt.plot(time_points.squeeze().cpu().numpy(), sample, 'r-', alpha=0.2)
            
            # Compute mean and std
            samples_array = np.array(samples)
            mean = np.mean(samples_array, axis=0)
            std = np.std(samples_array, axis=0)
            
            # Plot mean and confidence bands
            plt.plot(time_points.squeeze().cpu().numpy(), mean, 'r-', linewidth=2, label='Mean Reconstruction')
            plt.fill_between(
                time_points.squeeze().cpu().numpy(),
                mean - 2*std,
                mean + 2*std,
                color='r', alpha=0.2, label='±2σ'
            )
            
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.title(f'Example {i+1}: NF-Generated Samples')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"nf_samples_example_{i+1}.png"))
            plt.close()
    
    # Generate "morphing" between two trajectories
    with torch.no_grad():
        if batch_size >= 2:
            # Get latent codes of first two examples
            mu1, _ = model.encode(examples[0:1])
            mu2, _ = model.encode(examples[1:1])
            
            # Generate a path in the latent space between these two points
            num_steps = 10
            alphas = np.linspace(0, 1, num_steps)
            latent_path = []
            
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                latent_path.append(z)
            
            latent_path = torch.cat(latent_path, dim=0)
            
            # Decode the path
            trajectories = model.decode(latent_path, time_points)
            
            # Plot the morphing
            plt.figure(figsize=(12, 8))
            
            # Plot original trajectories
            plt.plot(time_points.squeeze().cpu().numpy(), examples[0].cpu().numpy(), 'b-', linewidth=2, label='Trajectory 1')
            plt.plot(time_points.squeeze().cpu().numpy(), examples[1].cpu().numpy(), 'g-', linewidth=2, label='Trajectory 2')
            
            # Plot interpolated trajectories
            time_np = time_points.squeeze().cpu().numpy()
            for i in range(num_steps):
                color = (1 - alphas[i], 0, alphas[i])  # Gradient from blue to green
                plt.plot(time_np, trajectories[i].cpu().numpy(), color=color, alpha=0.5)
            
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.title('Latent Space Interpolation Between Trajectories')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "nf_trajectory_morphing.png"))
            plt.close()
    
    # Visualize the effect of each flow layer on a trajectory
    with torch.no_grad():
        # Take the first example
        example = examples[0:1]
        
        # Encode to get latent
        z, _ = model.encode(example)
        
        # Apply each flow layer sequentially and visualize
        plt.figure(figsize=(12, 8))
        
        # Plot original trajectory
        plt.plot(time_points.squeeze().cpu().numpy(), example.squeeze().cpu().numpy(), 'k-', linewidth=2, label='Original')
        
        # Decode using the basis and modal nets directly without flows
        # We need to modify the forward pass for this
        if hasattr(model.decoder, 'basis_net') and hasattr(model.decoder, 'modal_net'):
            psi = model.decoder.basis_net(z)
            c = model.decoder.modal_net(time_points)
            direct_recon = torch.matmul(psi, c.transpose(0, 1))
            
            plt.plot(
                time_points.squeeze().cpu().numpy(), 
                direct_recon.squeeze().cpu().numpy(), 
                'b--', linewidth=2, label='Without Flows'
            )
        
        # Apply flows incrementally
        z_trans = z
        colors = ['g', 'b', 'r', 'c', 'm', 'y']  # Colors for different flow layers
        
        for i, flow in enumerate(model.decoder.flows):
            # Apply this flow
            z_trans, _ = flow(z_trans)
            
            # Decode
            psi = model.decoder.basis_net(z_trans)
            c = model.decoder.modal_net(time_points)
            recon = torch.matmul(psi, c.transpose(0, 1))
            
            color = colors[i % len(colors)]
            plt.plot(
                time_points.squeeze().cpu().numpy(),
                recon.squeeze().cpu().numpy(),
                f'{color}-', linewidth=2,
                label=f'After Flow {i+1}'
            )
        
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Effect of Flow Layers on Reconstruction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "nf_flow_layer_effect.png"))
        plt.close()

if __name__ == "__main__":
    main()