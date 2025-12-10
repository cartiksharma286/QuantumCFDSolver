import numpy as np

def naca4_half_thickness(x, t):
    """
    Calculate half-thickness yt of a symmetrical NACA 4-digit airfoil.
    x: Coordinate along chord (0 to 1)
    t: Maximum thickness as fraction of chord (e.g., 0.12 for NACA 0012)
    """
    term1 = 0.2969 * np.sqrt(x)
    term2 = -0.1260 * x
    term3 = -0.3516 * x**2
    term4 = 0.2843 * x**3
    term5 = -0.1015 * x**4
    return 5 * t * (term1 + term2 + term3 + term4 + term5)

def generate_naca_airfoil(shape, code='0012', chord_length=0.6, angle_of_attack=0.0):
    """
    Generate a boolean mask for a NACA airfoil extruded in Z (and W).
    
    Args:
        shape (tuple): (nw, nz, ny, nx) or (nz, ny, nx)
        code (str): 4-digit NACA code (e.g., '0012')
        chord_length (float): Fraction of domain length (0 to 1)
        angle_of_attack (float): Angle in degrees (rotation around leading edge or center)
    
    Returns:
        np.ndarray: Boolean mask (True = Obstacle/Blade)
    """
    # Parse 4-digit code (Assuming symmetric '00XX' for simplicity first)
    # First digit: max camber
    # Second digit: position of max camber
    # Last two: thickness
    
    thickness = int(code[2:]) / 100.0
    
    dims = len(shape)
    # Assume last two dims are y, x
    ny, nx = shape[-2], shape[-1]
    
    # Create coordinate grid
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Center the airfoil
    x_center = 0.5
    y_center = 0.5
    
    # Coordinate transformation for Angle of Attack
    rad = np.radians(angle_of_attack)
    # Shift to Origin
    X_s = X - x_center
    Y_s = Y - y_center
    # Rotate
    X_r = X_s * np.cos(rad) + Y_s * np.sin(rad)
    Y_r = -X_s * np.sin(rad) + Y_s * np.cos(rad)
    # Shift back effective coordinate for thickness check
    # But standard formula assumes leading edge at 0. Let's map X_r to [0, chord]
    
    # Simplified placement: Leading edge at x_center - chord/2
    le_x = -chord_length / 2.0
    te_x = chord_length / 2.0
    
    obstacle_2d = np.zeros((ny, nx), dtype=bool)
    
    # Points inside airfoil
    # 0 <= position_along_chord <= 1
    # position_along_chord = (X_r - le_x) / chord_length
    
    pos = (X_r - le_x) / chord_length
    
    # Filter points within chord range
    mask_range = (pos >= 0) & (pos <= 1)
    
    yt = naca4_half_thickness(pos[mask_range], thickness)
    
    # Check thickness condition: |Y_r| <= yt
    obstacle_2d[mask_range] = (np.abs(Y_r[mask_range]) <= yt)
    
    # Extrude to higher dimensions
    # If 3D: (nz, ny, nx)
    # If 4D: (nw, nz, ny, nx)
    
    if dims == 3:
        nz = shape[0]
        # Copy 2D mask to all Z
        # Or maybe partial span? Let's do full span.
        obstacle = np.tile(obstacle_2d, (nz, 1, 1))
    elif dims == 4:
        nw, nz = shape[0], shape[1]
        obstacle = np.tile(obstacle_2d, (nw, nz, 1, 1))
    else:
        obstacle = obstacle_2d
        
    return obstacle

if __name__ == "__main__":
    mask = generate_naca_airfoil((16, 32, 32), '0020', angle_of_attack=15)
    print(f"Generated mask with {np.sum(mask)} voxels")
