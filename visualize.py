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
    Plot a single frame of the simulation.
    """
    ny, nx = u.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    velocity_magnitude = np.sqrt(u**2 + v**2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Velocity Magnitude as contour/heatmap
    strm = ax.streamplot(X, Y, u, v, color='k', linewidth=0.5, density=1.0)
    cf = ax.contourf(X, Y, velocity_magnitude, levels=50, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Velocity Magnitude')
    
    ax.set_title(f"Quantum CFD - Step {step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if save_static:
        filename = os.path.join(output_dir, f"frame_{step:04d}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved {filename}")
        
    plt.close(fig)
    return fig

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Quantum CFD Results")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory containing .npy files")
    parser.add_argument("--animate", action="store_true", help="Create an animation (GIF)")
    parser.add_argument("--frames", action="store_true", help="Save individual frames as PNG")
    
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
        
    if not args.frames and not args.animate:
        print("No action selected. Use --animate or --frames.")
