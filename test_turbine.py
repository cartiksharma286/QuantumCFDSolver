import requests
import time
import os

BASE_URL = "http://localhost:8080"

def test_turbine_sim():
    print("Testing Turbine Simulation API...")
    
    config = {
        "naca_code": "0024", # Thick airfoil
        "angle_of_attack": 15.0, # High angle
        "steps": 20, # Short run
        "grid_size": 24, # Small grid
        "reynolds": 500.0,
        "forcing": True
    }
    
    # 1. Start Simulation
    try:
        resp = requests.post(f"{BASE_URL}/simulate", json=config)
        resp.raise_for_status()
        data = resp.json()
        sim_id = data["simulation_id"]
        print(f"Simulation Queued: {sim_id}")
    except Exception as e:
        print(f"Failed to start simulation: {e}")
        return

    # 2. Poll for Results
    print("Waiting for completion...")
    for i in range(20):
        time.sleep(2)
        resp = requests.get(f"{BASE_URL}/results/{sim_id}")
        if resp.status_code == 200:
            res = resp.json()
            files = res["files"]
            if "flow.gif" in files and "forces.png" in files:
                print("Simulation Completed Successfully!")
                print(f"Artifacts: {files}")
                return
            else:
                print(f"Progress... Files: {files}")
        else:
            print("Simulation not ready...")
            
    print("Timeout waiting for simulation.")

if __name__ == "__main__":
    test_turbine_sim()
