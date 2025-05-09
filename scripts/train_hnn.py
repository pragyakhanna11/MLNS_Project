#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the Physics-Informed VAE with Hamiltonian Neural Network encoder
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from data.loader import load_cell_migration_data
from models.composite.pi_hnn import PhysicsInformedHNN
from training.trainer import Trainer

def main():
    """Main function to train the HNN-based PIVAE model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Physics-Informed VAE with Hamiltonian Neural Network encoder"
    )
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to processed cell migration data")
    parser.add_argument("--output-dir", "-o", type=str, default="results/hnn_model",
                        help="Directory to save model and results")
    parser.add_argument("--window-size", "-w", type=int, default=20,
                        help="Window size for trajectory segments")
    parser.add_argument("--latent-dim", "-l", type=int, default=16,
                        help="Dimension of latent space")
    parser.add_argument("--hidden-dim", "-d", type=int, default=128,
                        help="Dimension of hidden layers")
    parser.add_argument("--output-dim", type=int, default=64,
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
    parser.add_argument("--physics-model", type=str, default="hamiltonian",
                        choices=["ou", "persistence", "hamiltonian"],
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
    model = PhysicsInformedHNN(
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
        model_type='hnn'
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
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write("Physics-Informed VAE with Hamiltonian Neural Network Training Configuration\n")
        f.write("===================================================================\n\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nFinal loss: {losses[-1]:.6f}\n")
        f.write(f"Final KL loss: {loss_history['kl'][-1]:.6f}\n")
        f.write(f"Final reconstruction loss: {loss_history['recon'][-1]:.6f}\n")
        f.write(f"Final physics loss: {loss_history['physics'][-1]:.6f}\n")
        if 'hamiltonian' in loss_history:
            f.write(f"Final Hamiltonian loss: {loss_history['hamiltonian'][-1]:.6f}\n")
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()