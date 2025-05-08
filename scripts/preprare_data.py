#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare cell migration data for Physics-Informed VAE models.
This script processes raw cell trajectory data, handles missing values,
normalizes the data, and saves it in a format ready for model training.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# Import from data module
from data.loader import load_cell_migration_data
from data.preprocessing import normalize_trajectories, handle_missing_data, filter_trajectories, compute_velocity_acceleration
from data.visualization import visualize_cell_trajectories, plot_trajectory_statistics, plot_velocity_acceleration, plot_phase_space

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare cell migration data for training")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to input cell migration data file")
    parser.add_argument("--output-dir", "-o", type=str, default="prepared_data",
                        help="Directory to save processed data and visualizations")
    parser.add_argument("--normalize", "-n", type=str, default="standard",
                        choices=["standard", "minmax", "robust", "none"],
                        help="Normalization method for trajectories")
    parser.add_argument("--missing-data", "-m", type=str, default="interpolate",
                        choices=["zero", "interpolate", "mask"],
                        help="Method to handle missing data")
    parser.add_argument("--filter", "-f", action="store_true",
                        help="Apply filtering to trajectories")
    parser.add_argument("--min-length", type=int, default=20,
                        help="Minimum length of valid trajectory segment for filtering")
    parser.add_argument("--max-displacement", type=float, default=30.0,
                        help="Maximum allowed displacement between timepoints for filtering")
    parser.add_argument("--window-size", "-w", type=int, default=20,
                        help="Window size for sliding window")
    parser.add_argument("--stride", "-s", type=int, default=5,
                        help="Stride for sliding window")
    parser.add_argument("--num-viz", type=int, default=5,
                        help="Number of trajectories to visualize")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time step size (in normalized units)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load raw data
    print(f"Loading cell migration data from {args.input}...")
    
    # Prepare filtering criteria if enabled
    filter_criteria = None
    if args.filter:
        filter_criteria = {
            'min_length': args.min_length,
            'max_displacement': args.max_displacement
        }
    
    # Load and preprocess data
    normalize = args.normalize != "none"
    cell_trajectories, num_timesteps = load_cell_migration_data(
        args.input,
        normalize=normalize,
        filter_criteria=filter_criteria
    )
    
    # Apply specific normalization if requested
    if normalize:
        print(f"Applying {args.normalize} normalization...")
        cell_trajectories = normalize_trajectories(cell_trajectories, normalization=args.normalize)
    
    # Handle missing data
    print(f"Handling missing data using {args.missing_data} method...")
    cell_trajectories = handle_missing_data(cell_trajectories, method=args.missing_data)
    
    # Save processed data
    processed_data_path = os.path.join(args.output_dir, "processed_trajectories.npy")
    np.save(processed_data_path, cell_trajectories)
    print(f"Saved processed data to {processed_data_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Visualize subset of trajectories
    viz_path = os.path.join(args.output_dir, "trajectory_visualization.png")
    visualize_cell_trajectories(
        cell_trajectories,
        num_cells=args.num_viz,
        title="Processed Cell Trajectories",
        save_path=viz_path
    )
    print(f"Saved trajectory visualization to {viz_path}")
    
    # 2. Plot trajectory statistics
    stats_path = os.path.join(args.output_dir, "trajectory_statistics.png")
    plot_trajectory_statistics(
        cell_trajectories,
        save_path=stats_path
    )
    print(f"Saved trajectory statistics to {stats_path}")
    
    # 3. Compute and visualize derivatives
    print("Computing velocity and acceleration...")
    velocity, acceleration = compute_velocity_acceleration(
        cell_trajectories,
        dt=args.dt,
        smooth=True
    )
    
    # Save derivatives
    np.save(os.path.join(args.output_dir, "velocity.npy"), velocity)
    np.save(os.path.join(args.output_dir, "acceleration.npy"), acceleration)
    
    # Visualize derivatives for a sample cell
    for i in range(min(3, cell_trajectories.shape[0])):
        # Position, velocity, acceleration plot
        deriv_path = os.path.join(args.output_dir, f"derivatives_cell_{i}.png")
        plot_velocity_acceleration(
            cell_trajectories, velocity, acceleration,
            cell_idx=i,
            save_path=deriv_path
        )
        
        # Phase space plot
        phase_path = os.path.join(args.output_dir, f"phase_space_cell_{i}.png")
        plot_phase_space(
            velocity, acceleration,
            cell_idx=i,
            save_path=phase_path
        )
    
    # 4. Create training data using sliding windows
    print(f"Creating training data with window size {args.window_size} and stride {args.stride}...")
    from data.loader import create_training_data
    X, Y, time_indices = create_training_data(
        cell_trajectories,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Save training data
    np.save(os.path.join(args.output_dir, "X_training_windows.npy"), X)
    np.save(os.path.join(args.output_dir, "Y_training_windows.npy"), Y)
    np.save(os.path.join(args.output_dir, "time_indices.npy"), time_indices)
    
    # Print summary statistics
    print("\nData Preparation Summary:")
    print(f"  Raw data shape: {cell_trajectories.shape}")
    print(f"  Number of cell trajectories: {cell_trajectories.shape[0]}")
    print(f"  Timesteps per trajectory: {cell_trajectories.shape[1]}")
    print(f"  Training windows: {X.shape[0]}")
    print(f"  Window size: {X.shape[1]}")
    
    # Create a metadata file with information about the preprocessing
    metadata_path = os.path.join(args.output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("Cell Migration Data Preprocessing Metadata\n")
        f.write("=========================================\n\n")
        f.write(f"Original data file: {args.input}\n")
        f.write(f"Normalization method: {args.normalize}\n")
        f.write(f"Missing data handling: {args.missing_data}\n")
        f.write(f"Filtering applied: {args.filter}\n")
        if args.filter:
            f.write(f"  Minimum trajectory length: {args.min_length}\n")
            f.write(f"  Maximum displacement: {args.max_displacement}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Stride: {args.stride}\n")
        f.write(f"Time step (dt): {args.dt}\n")
        f.write(f"\nProcessed data shape: {cell_trajectories.shape}\n")
        f.write(f"Training windows shape: {X.shape}\n")
    
    print(f"Saved preprocessing metadata to {metadata_path}")
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()