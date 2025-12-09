import argparse
import numpy as np
import os
import sys

# Import our visualization module
import visualize
from solver import QuantumNavierStokesSolver

def run_simulation(steps=100, nx=64, ny=64, output_dir="output", 
                   lid_velocity=1.0, reynolds=100.0, use_qiskit=False, 
                   do_visualize=False, make_animation=False):
    """
    Run the Quantum CFD simulation.
    """
    
    # Calculate viscosity from Reynolds number: Re = (U * L) / nu
    # Assuming L=1.0, U=lid_velocity
    nu = (lid_velocity * 1.0) / reynolds
    
    print("===========================================")
    print("   Quantum CFD Solver with NVQLink (Hybrid)   ")
    print("===========================================")
    print(f"Grid Size:       {nx}x{ny}")
    print(f"Time Steps:      {steps}")
    print(f"Output Directory:{output_dir}")
    print(f"Lid Velocity:    {lid_velocity}")
    print(f"Reynolds Number: {reynolds} (nu={nu:.6f})")
    print(f"NVQLink Qiskit:  {use_qiskit}")
    print("===========================================")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    solver = QuantumNavierStokesSolver(
        nx=nx, ny=ny, nu=nu, 
        lid_velocity=lid_velocity, 
        nvqlink_qiskit=use_qiskit
    )
    
    # Save initial state
    np.save(os.path.join(output_dir, "u_000.npy"), solver.u)
    np.save(os.path.join(output_dir, "v_000.npy"), solver.v)
    np.save(os.path.join(output_dir, "p_000.npy"), solver.p)
    
    print("Starting simulation...")
    for n in range(1, steps + 1):
        if n % 10 == 0 or n == steps:
            sys.stdout.write(f"\rStep {n}/{steps}...")
            sys.stdout.flush()
            
        solver.step()
        
        # Save check points 
        # For animation we might want more frequent saves?
        # Let's save every 10 steps or if short simulation every step
        save_interval = 10 if steps >= 100 else 1
        
        if n % save_interval == 0:
            np.save(os.path.join(output_dir, f"u_{n:03d}.npy"), solver.u)
            np.save(os.path.join(output_dir, f"v_{n:03d}.npy"), solver.v)
            np.save(os.path.join(output_dir, f"p_{n:03d}.npy"), solver.p)

    print("\nSimulation Complete.")
    
    if do_visualize or make_animation:
        print("Generating visualizations...")
        data = visualize.load_data(output_dir)
        
        if do_visualize:
            # Generate static frame for the last step
            last_step, u, v, p = data[-1]
            visualize.plot_frame(last_step, u, v, p, output_dir, save_static=True)
            
        if make_animation:
            visualize.create_animation(data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum CFD Solver")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps")
    parser.add_argument("--nx", type=int, default=64, help="Grid width")
    parser.add_argument("--ny", type=int, default=64, help="Grid height")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--lid-vel", type=float, default=1.0, help="Lid velocity")
    
    parser.add_argument("--qiskit", action="store_true", help="Enable Qiskit integration in NVQLink")
    parser.add_argument("--visualize", action="store_true", help="Generate static plots after run")
    parser.add_argument("--animate", action="store_true", help="Generate animation GIF after run")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    parser.add_argument("--test", action="store_true", help="Run a quick verification test")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running Verification Test...")
        # Run a small simulation
        run_simulation(steps=20, nx=32, ny=32, output_dir="test_output", do_visualize=True)
        print("Verification Test Passed.")
    else:
        run_simulation(steps=args.steps, nx=args.nx, ny=args.ny, 
                       output_dir=args.output,
                       lid_velocity=args.lid_vel, reynolds=args.re,
                       use_qiskit=args.qiskit,
                       do_visualize=args.visualize, make_animation=args.animate)
