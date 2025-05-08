import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

def visualize_cell_trajectories(cell_trajectories, num_cells=5, title=None, save_path=None, figsize=(12, 8)):
    """
    Visualize a subset of cell trajectories
    
    Args:
        cell_trajectories: numpy array of shape [num_cells, num_timesteps]
        num_cells: Number of cells to visualize
        title: Optional title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Select random cells if there are more than num_cells
    if cell_trajectories.shape[0] > num_cells:
        cell_indices = np.random.choice(cell_trajectories.shape[0], num_cells, replace=False)
    else:
        cell_indices = range(cell_trajectories.shape[0])
    
    for i, idx in enumerate(cell_indices):
        trajectory = cell_trajectories[idx]
        # Filter out zeros and NaNs
        valid_idx = np.where(~np.isnan(trajectory) & (trajectory != 0))[0]
        if len(valid_idx) > 10:  # Only plot if enough valid points
            plt.plot(valid_idx, trajectory[valid_idx], '-o', markersize=3, 
                     alpha=0.7, label=f'Cell {idx+1}')
    
    plt.xlabel('Time Step (15 min intervals)', fontsize=12)
    plt.ylabel('Position (μm)', fontsize=12)
    plt.title(title or 'Cell Migration Trajectories', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trajectory_statistics(cell_trajectories, save_path=None, figsize=(15, 10)):
    """
    Plot statistics of cell trajectories
    
    Args:
        cell_trajectories: numpy array of shape [num_cells, num_timesteps]
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate statistics
    num_cells, num_timesteps = cell_trajectories.shape
    
    # Mean and standard deviation over time
    valid_mask = ~np.isnan(cell_trajectories) & (cell_trajectories != 0)
    mean_position = np.zeros(num_timesteps)
    std_position = np.zeros(num_timesteps)
    
    for t in range(num_timesteps):
        valid_cells = valid_mask[:, t]
        if np.sum(valid_cells) > 0:
            mean_position[t] = np.mean(cell_trajectories[valid_cells, t])
            std_position[t] = np.std(cell_trajectories[valid_cells, t])
    
    # Plot 1: Mean trajectory with standard deviation
    axs[0, 0].plot(mean_position, 'b-', label='Mean Position')
    axs[0, 0].fill_between(range(num_timesteps), 
                         mean_position - std_position,
                         mean_position + std_position,
                         alpha=0.3, color='b', label='±1σ')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Position (μm)')
    axs[0, 0].set_title('Mean Cell Position Over Time')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Compute displacements
    displacements = []
    for i in range(num_cells):
        valid_idx = np.where(valid_mask[i, :])[0]
        if len(valid_idx) > 1:
            cell_displacements = np.diff(cell_trajectories[i, valid_idx])
            displacements.extend(cell_displacements)
    
    # Plot 2: Histogram of displacements
    axs[0, 1].hist(displacements, bins=30, alpha=0.7, density=True)
    axs[0, 1].set_xlabel('Displacement (μm)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].set_title('Distribution of Displacements')
    
    # Fit normal distribution to displacements
    if len(displacements) > 0:
        mean_disp = np.mean(displacements)
        std_disp = np.std(displacements)
        x = np.linspace(min(displacements), max(displacements), 100)
        y = 1/(std_disp * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_disp)**2 / (2 * std_disp**2))
        axs[0, 1].plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mean_disp:.2f}, σ={std_disp:.2f})')
        axs[0, 1].legend()
    
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: KDE of start and end positions
    valid_starts = []
    valid_ends = []
    
    for i in range(num_cells):
        valid_idx = np.where(valid_mask[i, :])[0]
        if len(valid_idx) > 0:
            valid_starts.append(cell_trajectories[i, valid_idx[0]])
            valid_ends.append(cell_trajectories[i, valid_idx[-1]])
    
    if len(valid_starts) > 1 and len(valid_ends) > 1:
        # KDE for start positions
        kde_start = gaussian_kde(valid_starts)
        x_start = np.linspace(min(valid_starts), max(valid_starts), 100)
        axs[1, 0].plot(x_start, kde_start(x_start), 'g-', linewidth=2, label='Start Positions')
        
        # KDE for end positions
        kde_end = gaussian_kde(valid_ends)
        x_end = np.linspace(min(valid_ends), max(valid_ends), 100)
        axs[1, 0].plot(x_end, kde_end(x_end), 'r-', linewidth=2, label='End Positions')
        
        axs[1, 0].set_xlabel('Position (μm)')
        axs[1, 0].set_ylabel('Density')
        axs[1, 0].set_title('Distribution of Start and End Positions')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Compute and visualize trajectory length
    traj_lengths = []
    for i in range(num_cells):
        valid_idx = np.where(valid_mask[i, :])[0]
        if len(valid_idx) > 1:
            # Total path length
            traj_lengths.append(np.sum(np.abs(np.diff(cell_trajectories[i, valid_idx]))))
    
    if len(traj_lengths) > 0:
        axs[1, 1].hist(traj_lengths, bins=20, alpha=0.7)
        axs[1, 1].set_xlabel('Trajectory Length (μm)')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].set_title('Distribution of Trajectory Lengths')
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_velocity_acceleration(trajectories, velocity, acceleration, 
                              cell_idx=0, save_path=None, figsize=(15, 5)):
    """
    Plot position, velocity, and acceleration for a single cell
    
    Args:
        trajectories: Position data [num_cells, num_timesteps]
        velocity: Velocity data [num_cells, num_timesteps]
        acceleration: Acceleration data [num_cells, num_timesteps]
        cell_idx: Index of cell to plot
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Get valid indices
    valid_mask = ~np.isnan(trajectories[cell_idx, :]) & (trajectories[cell_idx, :] != 0)
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) > 1:
        # Plot position
        axs[0].plot(valid_idx, trajectories[cell_idx, valid_idx], 'b-o', markersize=3)
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Position (μm)')
        axs[0].set_title('Position')
        axs[0].grid(True, alpha=0.3)
        
        # Plot velocity
        axs[1].plot(valid_idx, velocity[cell_idx, valid_idx], 'g-o', markersize=3)
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Velocity (μm/step)')
        axs[1].set_title('Velocity')
        axs[1].grid(True, alpha=0.3)
        
        # Plot acceleration
        axs[2].plot(valid_idx, acceleration[cell_idx, valid_idx], 'r-o', markersize=3)
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Acceleration (μm/step²)')
        axs[2].set_title('Acceleration')
        axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_phase_space(velocity, acceleration, cell_idx=0, save_path=None, figsize=(10, 8)):
    """
    Plot phase space (velocity vs. acceleration) for a single cell
    
    Args:
        velocity: Velocity data [num_cells, num_timesteps]
        acceleration: Acceleration data [num_cells, num_timesteps]
        cell_idx: Index of cell to plot
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    # Get valid indices
    valid_mask = ~np.isnan(velocity[cell_idx, :]) & (velocity[cell_idx, :] != 0) & \
                ~np.isnan(acceleration[cell_idx, :]) & (acceleration[cell_idx, :] != 0)
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) > 1:
        plt.figure(figsize=figsize)
        
        # Simple scatter plot
        plt.scatter(velocity[cell_idx, valid_idx], acceleration[cell_idx, valid_idx], 
                   alpha=0.6, c=valid_idx, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Time Step')
        
        # Add trajectory (connect dots with lines)
        plt.plot(velocity[cell_idx, valid_idx], acceleration[cell_idx, valid_idx], 
                'k-', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Velocity (μm/step)')
        plt.ylabel('Acceleration (μm/step²)')
        plt.title(f'Phase Space Trajectory for Cell {cell_idx}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()