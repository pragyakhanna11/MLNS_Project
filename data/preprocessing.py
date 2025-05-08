import numpy as np
from scipy import interpolate

def normalize_trajectories(trajectories, normalization='standard'):
    """
    Normalize cell trajectories
    
    Args:
        trajectories: numpy array of shape [num_cells, num_timesteps]
        normalization: Normalization method ('standard', 'minmax', or 'robust')
    
    Returns:
        normalized_trajectories: Normalized trajectories
    """
    normalized_trajectories = trajectories.copy()
    
    # Normalize each trajectory independently
    for i in range(trajectories.shape[0]):
        # Get valid indices (non-zero, non-NaN)
        valid_idx = np.where(~np.isnan(trajectories[i, :]) & (trajectories[i, :] != 0))[0]
        
        if len(valid_idx) > 1:  # Need at least 2 points to normalize
            if normalization == 'standard':
                # Standard normalization (zero mean, unit variance)
                mean = np.mean(trajectories[i, valid_idx])
                std = np.std(trajectories[i, valid_idx])
                if std > 0:
                    normalized_trajectories[i, valid_idx] = (trajectories[i, valid_idx] - mean) / std
            
            elif normalization == 'minmax':
                # Min-max normalization to [0, 1]
                min_val = np.min(trajectories[i, valid_idx])
                max_val = np.max(trajectories[i, valid_idx])
                if max_val > min_val:
                    normalized_trajectories[i, valid_idx] = (trajectories[i, valid_idx] - min_val) / (max_val - min_val)
            
            elif normalization == 'robust':
                # Robust normalization using median and IQR
                median = np.median(trajectories[i, valid_idx])
                q75, q25 = np.percentile(trajectories[i, valid_idx], [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    normalized_trajectories[i, valid_idx] = (trajectories[i, valid_idx] - median) / iqr
    
    return normalized_trajectories

def handle_missing_data(trajectories, method='zero'):
    """
    Handle missing data (NaN values) in cell trajectories
    
    Args:
        trajectories: numpy array of shape [num_cells, num_timesteps]
        method: Method to handle missing data ('zero', 'interpolate', or 'mask')
    
    Returns:
        processed_trajectories: Trajectories with missing data handled
    """
    processed_trajectories = trajectories.copy()
    
    if method == 'zero':
        # Replace NaN with zeros
        processed_trajectories = np.nan_to_num(processed_trajectories, nan=0.0)
    
    elif method == 'interpolate':
        # Interpolate missing values
        for i in range(trajectories.shape[0]):
            # Get valid indices (non-NaN)
            valid_idx = np.where(~np.isnan(trajectories[i, :]))[0]
            if len(valid_idx) > 1:  # Need at least 2 points to interpolate
                # Create interpolation function
                f = interpolate.interp1d(
                    valid_idx, trajectories[i, valid_idx],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                
                # Get indices of NaN values within the range of valid indices
                nan_idx = np.where(np.isnan(trajectories[i, :]))[0]
                interp_idx = nan_idx[(nan_idx >= valid_idx[0]) & (nan_idx <= valid_idx[-1])]
                
                if len(interp_idx) > 0:
                    # Interpolate missing values
                    processed_trajectories[i, interp_idx] = f(interp_idx)
                
        # Replace any remaining NaN with zeros
        processed_trajectories = np.nan_to_num(processed_trajectories, nan=0.0)
    
    elif method == 'mask':
        # Keep NaN values (to be used with masking during training)
        pass
    
    return processed_trajectories

def filter_trajectories(trajectories, criteria):
    """
    Filter cell trajectories based on criteria
    
    Args:
        trajectories: numpy array of shape [num_cells, num_timesteps]
        criteria: Dictionary of filtering criteria
            - min_length: Minimum length of valid trajectory segment
            - max_displacement: Maximum allowed displacement between timepoints
            - min_total_displacement: Minimum total displacement
    
    Returns:
        filtered_trajectories: Filtered trajectories
    """
    num_cells = trajectories.shape[0]
    include_cell = np.ones(num_cells, dtype=bool)
    
    for i in range(num_cells):
        # Get valid indices (non-zero, non-NaN)
        valid_idx = np.where(~np.isnan(trajectories[i, :]) & (trajectories[i, :] != 0))[0]
        
        # Check minimum length
        if 'min_length' in criteria and len(valid_idx) < criteria['min_length']:
            include_cell[i] = False
            continue
        
        if len(valid_idx) > 1:
            # Calculate displacements
            displacements = np.abs(np.diff(trajectories[i, valid_idx]))
            
            # Check maximum displacement
            if 'max_displacement' in criteria and np.max(displacements) > criteria['max_displacement']:
                include_cell[i] = False
                continue
            
            # Check minimum total displacement
            if 'min_total_displacement' in criteria and np.sum(displacements) < criteria['min_total_displacement']:
                include_cell[i] = False
                continue
    
    # Return filtered trajectories
    return trajectories[include_cell, :]

def compute_velocity_acceleration(trajectories, dt=1.0, smooth=True, window=3):
    """
    Compute velocity and acceleration from position data
    
    Args:
        trajectories: numpy array of shape [num_cells, num_timesteps]
        dt: Time step size
        smooth: Whether to smooth the data before derivatives
        window: Window size for smoothing
    
    Returns:
        velocity: Velocity trajectories
        acceleration: Acceleration trajectories
    """
    num_cells, num_timesteps = trajectories.shape
    velocity = np.zeros_like(trajectories)
    acceleration = np.zeros_like(trajectories)
    
    for i in range(num_cells):
        # Get valid indices (non-zero, non-NaN)
        valid_idx = np.where(~np.isnan(trajectories[i, :]) & (trajectories[i, :] != 0))[0]
        
        if len(valid_idx) > 2:  # Need at least 3 points for acceleration
            # Get trajectory segment
            traj_segment = trajectories[i, valid_idx]
            
            # Apply smoothing if requested
            if smooth and len(traj_segment) > window:
                # Simple moving average
                kernel = np.ones(window) / window
                traj_segment = np.convolve(traj_segment, kernel, mode='same')
                
                # Fix endpoints (which are affected by boundary effects)
                half_window = window // 2
                if half_window > 0:
                    traj_segment[:half_window] = trajectories[i, valid_idx[:half_window]]
                    traj_segment[-half_window:] = trajectories[i, valid_idx[-half_window:]]
            
            # Compute velocity (central difference)
            v = np.zeros_like(traj_segment)
            v[1:-1] = (traj_segment[2:] - traj_segment[:-2]) / (2 * dt)
            
            # Forward/backward difference at endpoints
            v[0] = (traj_segment[1] - traj_segment[0]) / dt
            v[-1] = (traj_segment[-1] - traj_segment[-2]) / dt
            
            # Compute acceleration (central difference)
            a = np.zeros_like(traj_segment)
            a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
            
            # Forward/backward difference at endpoints
            a[0] = (v[1] - v[0]) / dt
            a[-1] = (v[-1] - v[-2]) / dt
            
            # Store results
            velocity[i, valid_idx] = v
            acceleration[i, valid_idx] = a
    
    return velocity, acceleration