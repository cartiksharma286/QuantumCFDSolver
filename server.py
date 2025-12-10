from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from solver import QuantumHyperFluidSolver
from geometry_loader import generate_naca_airfoil

app = FastAPI(title="Quantum CFD Turbine Server")

OUTPUT_DIR = "simulation_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class SimulationConfig(BaseModel):
    naca_code: str = "0012" # e.g. 0012, 2412
    angle_of_attack: float = 0.0
    steps: int = 50
    reynolds: float = 1000.0
    grid_size: int = 32 # assumes cubic nx=ny=nz
    forcing: bool = False
    forcing_intensity: float = 0.0

def run_simulation_task(sim_id: str, config: SimulationConfig):
    sim_dir = os.path.join(OUTPUT_DIR, sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    
    print(f"[{sim_id}] Starting Turbine Simulation (NACA {config.naca_code})")
    
    # 1. Generate Geometry
    # Shape: (nw=1, nz=grid, ny=grid, nx=grid) for 3D/pseudo-4D
    nw, nz, ny, nx = 3, config.grid_size, config.grid_size, config.grid_size
    shape = (nw, nz, ny, nx)
    
    # Generate mask for the blade
    print(f"[{sim_id}] Generating Airfoil...")
    obstacle_mask = generate_naca_airfoil(
        shape, 
        code=config.naca_code, 
        angle_of_attack=config.angle_of_attack,
        chord_length=0.4
    )
    
    # 2. Initialize Solver
    lid_vel = 1.0
    nu = (lid_vel * 1.0) / config.reynolds
    
    solver = QuantumHyperFluidSolver(
        nx=nx, ny=ny, nz=nz, nw=nw,
        nu=nu,
        lid_velocity=lid_vel,
        obstacles=obstacle_mask, # Pass mask
        forcing_intensity=config.forcing_intensity if config.forcing else 0.0
    )
    
    # 3. Time Loop
    lift_history = []
    drag_history = []
    
    frames = []
    
    print(f"[{sim_id}] Running {config.steps} steps...")
    for n in range(config.steps):
        solver.step(n)
        
        # Compute Forces (Quantum Surface Integral)
        forces = solver.compute_quantum_surface_integral()
        lift_history.append(forces['lift'])
        drag_history.append(forces['drag'])
        
        if n % 2 == 0:
            # Capture frame (Middle Z, Middle W)
            # Velocity magnitude
            mid_w = nw // 2
            mid_z = nz // 2
            u_slice = solver.u[mid_w, mid_z, :, :]
            v_slice = solver.v[mid_w, mid_z, :, :]
            mag = np.sqrt(u_slice**2 + v_slice**2)
            frames.append(mag.copy())

    # 4. Save Results
    # Plots
    plt.figure()
    plt.plot(lift_history, label='Lift')
    plt.plot(drag_history, label='Drag')
    plt.legend()
    plt.title(f"Aerodynamic Forces (NACA {config.naca_code}, Re={config.reynolds})")
    plt.savefig(os.path.join(sim_dir, "forces.png"))
    plt.close()
    
    # Animation
    if len(frames) > 0:
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0], animated=True, origin='lower', cmap='viridis')
        ax.set_title(f"Velocity Field (NACA {config.naca_code})")
        
        def update(frame):
            im.set_array(frame)
            return [im]
            
        anim = FuncAnimation(fig, update, frames=frames, blit=True)
        anim.save(os.path.join(sim_dir, "flow.gif"), writer='pillow', fps=10)
        plt.close()
        
    print(f"[{sim_id}] Simulation Complete.")

@app.post("/simulate")
async def start_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    sim_id = str(uuid.uuid4())
    background_tasks.add_task(run_simulation_task, sim_id, config)
    return {"simulation_id": sim_id, "status": "queued"}

@app.get("/results/{sim_id}")
async def get_results(sim_id: str):
    sim_dir = os.path.join(OUTPUT_DIR, sim_id)
    if not os.path.exists(sim_dir):
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    files = os.listdir(sim_dir)
    return {"simulation_id": sim_id, "files": files, "base_path": os.path.abspath(sim_dir)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
