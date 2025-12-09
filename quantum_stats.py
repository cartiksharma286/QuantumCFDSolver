import numpy as np

def fermi_dirac_distribution(energy, temperature, chemical_potential=0.0, k_b=1.0):
    """
    Calculate the Fermi-Dirac distribution.
    
    f(E) = 1 / (exp((E - mu) / (kB * T)) + 1)
    
    Args:
        energy (np.ndarray): Energy levels.
        temperature (float): Temperature (T). must be > 0.
        chemical_potential (float): Chemical potential (mu).
        k_b (float): Boltzmann constant.
        
    Returns:
        np.ndarray: Probability/Occupancy.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
        
    beta = 1.0 / (k_b * temperature)
    exponent = beta * (energy - chemical_potential)
    
    # Avoid overflow
    exponent = np.clip(exponent, -700, 700)
    
    return 1.0 / (np.exp(exponent) + 1.0)

def bose_einstein_distribution(energy, temperature, chemical_potential=0.0, k_b=1.0):
    """
    Calculate the Bose-Einstein distribution.
    
    f(E) = 1 / (exp((E - mu) / (kB * T)) - 1)
    
    Args:
        energy (np.ndarray): Energy levels.
        temperature (float): Temperature (T). must be > 0.
        chemical_potential (float): Chemical potential (mu).
        k_b (float): Boltzmann constant.
        
    Returns:
        np.ndarray: Probability/Occupancy.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
        
    beta = 1.0 / (k_b * temperature)
    exponent = beta * (energy - chemical_potential)
    
    # Avoid overflow
    exponent = np.clip(exponent, -700, 700)
    
    # For Bose-Einstein, term must be > 0, so exp(exponent) > 1 -> exponent > 0
    # Usually E > mu.
    
    den = np.exp(exponent) - 1.0
    
    # Handle singularity/numerical instability near 0
    with np.errstate(divide='ignore'):
        dist = 1.0 / den
        
    return dist

def initialize_field_quantum(shape, distribution_type='fermi-dirac', T=1.0, mu=0.5):
    """
    Initialize a 2D field based on a quantum statistical distribution.
    This creates an 'energy landscape' based on spatial position and maps it to a distribution.
    
    Args:
        shape (tuple): Shape of the grid (ny, nx).
        distribution_type (str): 'fermi-dirac' or 'bose-einstein'.
        T (float): Temperature parameter.
        mu (float): Chemical potential parameter.
        
    Returns:
        np.ndarray: The initialized field (e.g., density or temperature).
    """
    ny, nx = shape
    # Create a spatial energy gradient for demonstration
    y = np.linspace(0, 1, ny)
    x = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x, y)
    
    # Energy landscape: low in middle, high at edges (potential well)
    # E(x,y) = k * (x-0.5)^2 + (y-0.5)^2
    energy = 5.0 * ((X - 0.5)**2 + (Y - 0.5)**2)
    
    if distribution_type == 'fermi-dirac':
        return fermi_dirac_distribution(energy, T, mu)
    elif distribution_type == 'bose-einstein':
        return bose_einstein_distribution(energy + 0.1, T, mu) # Add offset to ensure E > mu
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
