import numpy as np
import torch
import os
from .preprocessing import normalize_trajectories, handle_missing_data, filter_trajectories

def load_cell_migration_data(file_path, max_cells=None, normalize=True, filter_criteria=None):
    """
    Load cell migration data from a text file
    
    Args:
        file_path: Path to the cell migration data file
        max_cells: Maximum number of cells to load (None for all)
        normalize: Whether to normalize the data
        filter_criteria: Dictionary of filtering criteria
    
    Returns:
        cell_trajectories: numpy array of shape [num_cells, num_timesteps]
        time_steps: number of time steps in each trajectory
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Load raw data
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading data file: {e}")
    
    # Determine if we need to limit number of cells
    if max_cells is not None and max_cells < data.shape[0]:
        data = data[:max_cells]
    
    # Handle missing data
    data = handle_missing_data(data)
    
    # Apply filtering if specified
    if filter_criteria is not None:
        data = filter_trajectories(data, filter_criteria)
    
    # Normalize trajectories if requested
    if normalize:
        data = normalize_trajectories(data)
    
    # Get dimensions
    num_cells, num_timesteps = data.shape
    
    return data, num_timesteps

def create_training_data(cell_trajectories, window_size=20, stride=5, min_valid_ratio=0.8):
    """
    Create training data by sliding windows over cell trajectories
    
    Args:
        cell_trajectories: numpy array of shape [num_cells, num_timesteps]
        window_size: Size of sliding window
        stride: Step size for sliding window
        min_valid_ratio: Minimum ratio of valid (non-zero, non-NaN) points required
    
    Returns:
        X: Input trajectory segments
        Y: Output trajectory segments (shifted by 1 time step)
        time_points: Corresponding time points for each window
    """
    num_cells, num_timesteps = cell_trajectories.shape
    X, Y = [], []
    time_indices = []
    
    for cell in range(num_cells):
        for t in range(0, num_timesteps - window_size - 1, stride):
            # Input window
            x_window = cell_trajectories[cell, t:t+window_size]
            # Output window (shifted by 1 for prediction)
            y_window = cell_trajectories[cell, t+1:t+window_size+1]
            
            # Skip windows with too many zeros or NaNs
            valid_count = np.count_nonzero(~np.isnan(x_window) & (x_window != 0))
            if valid_count >= window_size * min_valid_ratio:
                X.append(x_window)
                Y.append(y_window)
                time_indices.append(np.arange(t, t+window_size))
    
    # Convert to arrays
    X = np.array(X)
    Y = np.array(Y)
    time_indices = np.array(time_indices)
    
    return X, Y, time_indices

def load_dataset_as_tensors(file_path, window_size=20, stride=5, normalize=True, device="cpu"):
    """
    Load and prepare dataset as PyTorch tensors
    
    Args:
        file_path: Path to the cell migration data
        window_size: Window size for trajectory segments
        stride: Stride for sliding window
        normalize: Whether to normalize the data
        device: Device to place tensors on
    
    Returns:
        X_tensor: Input trajectory segments as tensor
        Y_tensor: Output trajectory segments as tensor
        time_tensor: Time points tensor
    """
    # Load raw data
    cell_trajectories, num_timesteps = load_cell_migration_data(
        file_path, normalize=normalize
    )
    
    # Create training data with sliding windows
    X, Y, time_indices = create_training_data(
        cell_trajectories, window_size=window_size, stride=stride
    )
    
    # Create normalized time points
    time_points = np.linspace(0, 1, window_size)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    time_tensor = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1).to(device)
    
    return X_tensor, Y_tensor, time_tensor, cell_trajectories