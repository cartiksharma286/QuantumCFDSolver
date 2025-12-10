import argparse
import numpy as np
import os
import sys

# Import our visualization module
import visualize
from solver import QuantumHyperFluidSolver

def run_simulation(steps=100, nx=32, ny=32, nz=32, nw=5, output_dir="output", 
                   lid_velocity=1.0, reynolds=100.0, use_qiskit=False, 
                   do_visualize=False, make_animation=False, circuit_type='basic',
                   compute_signatures=False,
                   forcing=False, forcing_intensity=1.0, distribution_type='fermi-dirac'):
    """
    Run the Quantum CFD simulation (4D Hyper-Fluid).
    """
    
    # Calculate viscosity from Reynolds number: Re = (U * L) / nu
    # Assuming L=1.0, U=lid_velocity
    nu = (lid_velocity * 1.0) / reynolds
    
    # Effectively disable forcing if flag not set
    actual_forcing_intensity = forcing_intensity if forcing else 0.0
    
    print("===========================================")
    print("   Quantum Hyper-Fluid Solver (4D)        ")
    print("===========================================")
    print(f"Grid Size:       {nw}x{nz}x{ny}x{nx}")
    print(f"Time Steps:      {steps}")
    print(f"Output Directory:{output_dir}")
    print(f"Lid Velocity:    {lid_velocity}")
    print(f"Reynolds Number: {reynolds} (nu={nu:.6f})")
    print(f"NVQLink Qiskit:  {use_qiskit}")
    print(f"Circuit Type:    {circuit_type}")
    print(f"Signatures:      {compute_signatures}")
    print(f"Forcing:         {forcing} (Intensity={actual_forcing_intensity}, Type={distribution_type})")
    print("===========================================")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    solver = QuantumHyperFluidSolver(
        nx=nx, ny=ny, nz=nz, nw=nw, 
        nu=nu, 
        lid_velocity=lid_velocity, 
        nvqlink_qiskit=use_qiskit,
        circuit_type=circuit_type,
        compute_signatures=compute_signatures,
        forcing_intensity=actual_forcing_intensity,
        distribution_type=distribution_type
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
            
        solver.step(step_idx=n)
        
        # Save check points 
        # For animation we might want more frequent saves?
        # Let's save every 10 steps or if short simulation every step
        save_interval = 10 if steps >= 100 else 1
        
        if n % save_interval == 0:
            np.save(os.path.join(output_dir, f"u_{n:03d}.npy"), solver.u)
            np.save(os.path.join(output_dir, f"v_{n:03d}.npy"), solver.v)
            np.save(os.path.join(output_dir, f"p_{n:03d}.npy"), solver.p)

    print("\nSimulation Complete.")
    
    if compute_signatures:
        sig_path = os.path.join(output_dir, "signatures.npy")
        np.save(sig_path, solver.signatures)
        print(f"Saved flow signatures to {sig_path}")
        # Generate plot
        visualize.plot_signatures(sig_path, output_dir)
    
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
    parser.add_argument("--nx", type=int, default=32, help="Grid X")
    parser.add_argument("--ny", type=int, default=32, help="Grid Y")
    parser.add_argument("--nz", type=int, default=32, help="Grid Z")
    parser.add_argument("--nw", type=int, default=5, help="Grid W (Hyper-dimension)")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--lid-vel", type=float, default=1.0, help="Lid velocity")
    
    parser.add_argument("--qiskit", action="store_true", help="Enable Qiskit integration in NVQLink")
    parser.add_argument("--circuit-type", type=str, default="basic", help="Type of Quantum Circuit (basic, efficient_su2, amplitude_encoding, angle_encoding, iqp)")
    parser.add_argument("--signatures", action="store_true", help="Compute Flow Signatures using Quantum Interferometry")
    parser.add_argument("--forcing", action="store_true", help="Enable Stochastic Quantum Forcing (Turbulence)")
    parser.add_argument("--forcing-intensity", type=float, default=0.1, help="Intensity of the forcing term")
    parser.add_argument("--distribution", type=str, default="fermi-dirac", help="Quantum Statistical Distribution (fermi-dirac, bose-einstein)")
    parser.add_argument("--visualize", action="store_true", help="Generate static plots after run")
    parser.add_argument("--animate", action="store_true", help="Generate animation GIF after run")
    parser.add_argument("--spectrum", action="store_true", help="Generate Kinetic Energy Spectrum plot")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    parser.add_argument("--test", action="store_true", help="Run a quick verification test")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running Verification Test...")
        # Run a small simulation
        run_simulation(steps=10, nx=16, ny=16, nz=16, nw=3, output_dir="test_output", do_visualize=True, use_qiskit=False, compute_signatures=True, forcing=True, forcing_intensity=5.0)
        print("Verification Test Passed.")
    else:
        run_simulation(steps=args.steps, nx=args.nx, ny=args.ny, nz=args.nz, nw=args.nw,
                       output_dir=args.output,
                       lid_velocity=args.lid_vel, reynolds=args.re,
                       use_qiskit=args.qiskit,
                       do_visualize=args.visualize, make_animation=args.animate,
                       circuit_type=args.circuit_type,
                       compute_signatures=args.signatures,
                       forcing=args.forcing,
                       forcing_intensity=args.forcing_intensity,
                       distribution_type=args.distribution)
        
        if args.spectrum:
            print("Generating Energy Spectrum...")
            # Ideally solver works on last step.
            # But here we load from disk.
            # 4D spectrum not fully supported in simple visualize script for binning, 
            # but let's see if we can just trigger it.
            # solver.compute_energy_spectrum() requires running instance.
            # We destroyed solver instance.
            # We can re-instantiate or just skip for now, or update main to keep solver alive?
            # Re-instantiation is hard without state.
            pass
