import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
import glob

def load_data(output_dir):
    """
    Load simulation data from the output directory.
    Returns lists of (step, u, v, p) tuples sorted by step.
    """
    u_files = sorted(glob.glob(os.path.join(output_dir, "u_*.npy")))
    v_files = sorted(glob.glob(os.path.join(output_dir, "v_*.npy")))
    p_files = sorted(glob.glob(os.path.join(output_dir, "p_*.npy")))
    
    data = []
    for u_f, v_f, p_f in zip(u_files, v_files, p_files):
        # Extract step number
        basename = os.path.basename(u_f)
        step = int(basename.split('_')[1].split('.')[0])
        
        u = np.load(u_f)
        v = np.load(v_f)
        p = np.load(p_f)
        data.append((step, u, v, p))
        
    return data

def plot_frame(step, u, v, p, output_dir, save_static=False):
    """
    Plot velocity magnitude and pressure field.
    Handles 2D data (ny, nx) or 4D data (nw, nz, ny, nx) via slicing.
    """
    if u.ndim == 4:
        # Slice the hyper-cube
        nw, nz, ny, nx = u.shape
        mw, mz = nw // 2, nz // 2
        u_slice = u[mw, mz, :, :]
        v_slice = v[mw, mz, :, :]
        p_slice = p[mw, mz, :, :]
        msg = f" (Slice 4D: w={mw}, z={mz})"
    else:
        u_slice = u
        v_slice = v
        p_slice = p
        msg = ""

    velocity = np.sqrt(u_slice**2 + v_slice**2)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pressure
    im0 = ax[0].imshow(p_slice, cmap='viridis', origin='lower')
    ax[0].set_title(f'Pressure Field {msg}')
    fig.colorbar(im0, ax=ax[0])
    
    # Velocity
    im1 = ax[1].imshow(velocity, cmap='plasma', origin='lower')
    ax[1].set_title(f'Velocity Magnitude {msg}')
    fig.colorbar(im1, ax=ax[1])
    
    plt.suptitle(f'Timestep {step}')
    
    path = os.path.join(output_dir, f'frame_{step:04d}.png')
    plt.savefig(path)
    plt.close()
    
    if save_static:
        print(f"Saved frame to {path}")

def create_animation(data, output_dir, fps=10):
    """
    Create an animation of the flow.
    """
    print("Creating animation...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial setup
    step0, u0, v0, p0 = data[0]
    ny, nx = u0.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Init plot objects
    contour = ax.contourf(X, Y, np.sqrt(u0**2 + v0**2), levels=50, cmap='viridis')
    # streamplot is hard to animate efficiently in matplotlib (it redraws everything), 
    # so we might use quiver or just redraw. 
    # For simplicity in 'FuncAnimation', clearing and redrawing is easiest but slower.
    
    def update(frame_idx):
        ax.clear()
        step, u, v, p = data[frame_idx]
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        ax.contourf(X, Y, velocity_magnitude, levels=50, cmap='viridis')
        # Reducing density of quiver for readability
        stride = 4
        ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
                  u[::stride, ::stride], v[::stride, ::stride], color='white')
        
        ax.set_title(f"Quantum CFD - Step {step}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=1000/fps)
    
    save_path = os.path.join(output_dir, "simulation_flow.gif")
    # Requires imagemagick or ffmpeg mostly. We'll try pillow for GIF which is standard.
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"Animation saved to {save_path}")

def plot_signatures(sig_file, output_dir):
    """
    Plot statistical flow signatures from Quantum Interferometry.
    1. Mean Fidelity vs Time.
    2. Histogram of Fidelity Distribution (Last Step).
    """
    try:
        data = np.load(sig_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"[Visualize] Signature file not found: {sig_file}")
        return

    steps = [d['step'] for d in data]
    means = [d['fid_mean'] for d in data]
    stds = [d['fid_std'] for d in data]
    
    # 1. Fidelity vs Time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(steps, means, yerr=stds, fmt='o-', capsize=5, label='Mean Fidelity')
    ax.set_title("Quantum Flow Fidelity (Temporal Autocorrelation)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Fidelity |<u(t)|u(t-1)>|^2")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    
    path_time = os.path.join(output_dir, "signature_fidelity_time.png")
    plt.savefig(path_time)
    plt.close()
    print(f"Saved fidelity time plot to {path_time}")
    
    # 2. Histogram (Last Step)
    if len(data) > 0:
        last_data = data[-1]
        hist_vals = last_data['histogram']
        step = last_data['step']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(hist_vals, bins=30, range=(0,1), density=True, alpha=0.7, color='purple')
        ax.set_title(f"Fidelity Distribution (Step {step})")
        ax.set_xlabel("Fidelity")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        
        path_hist = os.path.join(output_dir, f"signature_hist_{step:04d}.png")
        plt.savefig(path_hist)
        plt.close()
        print(f"Saved fidelity histogram to {path_hist}")

def plot_energy_spectrum(k, E, output_dir, step):
    """
    Plot kinetic energy spectrum E(k) vs k.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k, E, 'b-', label='Energy Spectrum')
    
    # Kolmogorov -5/3 line
    k_ref = k[len(k)//4 : 3*len(k)//4]
    if len(k_ref) > 0:
        E_ref = E[len(E)//2] * (k_ref / k[len(E)//2])**(-5/3)
        ax.loglog(k_ref, E_ref, 'k--', label='Kolmogorov -5/3')
        
    ax.set_title(f"Kinetic Energy Spectrum (Step {step})")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy E(k)")
    ax.legend()
    ax.grid(True, which="both", ls="-")
    
    path = os.path.join(output_dir, f'spectrum_{step:04d}.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved spectrum plot to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Quantum CFD Results")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory containing .npy files")
    parser.add_argument("--animate", action="store_true", help="Create an animation (GIF)")
    parser.add_argument("--frames", action="store_true", help="Save individual frames as PNG")
    parser.add_argument("--spectrum", action="store_true", help="Plot Energy Spectrum for last step")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} not found.")
        exit(1)
        
    data = load_data(args.output_dir)
    if not data:
        print("No data found to visualize.")
        exit(1)
        
    print(f"Loaded {len(data)} time steps.")
    
    if args.frames:
        for step, u, v, p in data:
            plot_frame(step, u, v, p, args.output_dir, save_static=True)
            
    if args.animate:
        create_animation(data, args.output_dir)
        
    if args.spectrum:
        # Load last step for spectrum
        if len(data) > 0:
            step, u, v, p = data[-1]
            
            # Compute spectrum
            # For 2D or 4D slice
            if u.ndim == 4:
                u = u[u.shape[0]//2, u.shape[1]//2, :, :]
                v = v[v.shape[0]//2, v.shape[1]//2, :, :]
            
            ft_u = np.fft.fft2(u)
            ft_v = np.fft.fft2(v)
            energy_spatial = 0.5 * (np.abs(ft_u)**2 + np.abs(ft_v)**2)
            ny, nx = energy_spatial.shape
            k_max = min(ny, nx) // 2
            k_bins = np.arange(0, k_max)
            energy_bins = np.zeros(len(k_bins))
            ky = np.fft.fftfreq(ny) * ny
            kx = np.fft.fftfreq(nx) * nx
            KX, KY = np.meshgrid(kx, ky)
            K = np.sqrt(KX**2 + KY**2)
            for i in range(1, len(k_bins)):
                indices = (K >= k_bins[i-1]) & (K < k_bins[i])
                if np.any(indices):
                    energy_bins[i] = np.sum(energy_spatial[indices])
            
            plot_energy_spectrum(k_bins[1:], energy_bins[1:], args.output_dir, step)

    # Check for signatures
    sig_file = os.path.join(args.output_dir, "signatures.npy")
    if os.path.exists(sig_file):
        print("Found signatures.npy, plotting...")
        plot_signatures(sig_file, args.output_dir)

    if not args.frames and not args.animate and not args.spectrum:
        print("No action selected. Use --animate, --frames, or --spectrum.")

