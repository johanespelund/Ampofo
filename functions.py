import numpy as np

def time_average(filepath, time_interval, patch_name, quantity):
    """
    Calculate the time average of a specified quantity over a given time interval for a specific patch.

    Parameters:
    filepath (str): Path to the data file.
    time_interval (tuple): A tuple containing the start and end times (start_time, end_time).
    patch_name (str): Name of the patch to filter (e.g., 'wall_right').
    quantity (str): The quantity to calculate the average for ('min', 'max', 'integral').

    Returns:
    float: The time-averaged value of the specified quantity.
    """
    
    # Mapping from quantity name to the respective column index
    quantity_map = {'min': 2, 'max': 3, 'integral': 4}
    
    # Validate quantity
    if quantity not in quantity_map:
        raise ValueError("Invalid quantity. Choose from 'min', 'max', 'integral'.")
    
    times = []
    values = []
    
    # Open and read the file
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split()
            
            if len(parts) != 5:  # Skip any malformed lines
                continue
            
            time = float(parts[0])
            patch = parts[1]
            value = float(parts[quantity_map[quantity]])
            
            # Filter by time interval and patch name
            if patch == patch_name and time_interval[0] <= time <= time_interval[1]:
                times.append(time)
                values.append(value)
    
    if not times:
        raise ValueError(f"No data found for patch '{patch_name}' in the given time interval.")
    
    # Convert to numpy arrays
    times = np.array(times)
    values = np.array(values)
    
    # Calculate the time average using the trapezoidal rule
    time_average_value = np.trapz(values, times) / (times[-1] - times[0])
    
    return time_average_value

