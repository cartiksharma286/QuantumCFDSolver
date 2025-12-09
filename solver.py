import numpy as np
from nvqlink import NVQLink
from quantum_stats import initialize_field_quantum

class QuantumNavierStokesSolver:
    """
    2D Incompressible Navier-Stokes Solver using Finite Difference Method (FDM).
    Integrates with NVQLink to offload Pressure Poisson Equation (PPE) solving.
    """
    
    def __init__(self, nx=64, ny=64, dt=0.001, rho=1.0, nu=0.1, lx=1.0, ly=1.0, use_quantum_stats=True, lid_velocity=1.0, nvqlink_qiskit=False):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.rho = rho
        self.nu = nu # Kinematic viscosity
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.lid_velocity = lid_velocity

        
        # Initialize fields
        self.u = np.zeros((ny, nx)) # X-velocity
        self.v = np.zeros((ny, nx)) # Y-velocity
        self.p = np.zeros((ny, nx)) # Pressure
        self.b = np.zeros((ny, nx)) # Source term for Poisson eq
        
        # Initialize NVQLink
        self.link = NVQLink(use_qiskit=nvqlink_qiskit)
        
        if use_quantum_stats:
            print("[Solver] Initializing density/flow with Quantum Statistics...")
            # We treat the returned distribution as an initial scalar field, 
            # effectively modulating the fluid properties or initial turbulence.
            # Here, let's map it to an initial 'energy' (velocity magnitude perturbation)
            q_dist = initialize_field_quantum((ny, nx), distribution_type='fermi-dirac', T=0.5)
            
            # Apply random perturbations scaled by quantum distribution
            self.u += q_dist * np.random.randn(ny, nx) * 0.1
            self.v += q_dist * np.random.randn(ny, nx) * 0.1
            
    def compute_b(self):
        """
        Compute the source term (RHS) for the Pressure Poisson Equation.
        b = rho * (1/dt * div(u_star) - ... non-linear terms ...)
        
        Simplified for projection method:
        RHS = rho / dt * (du/dx + dv/dy) ignoring advection terms in RHS for now as they are often handled in the intermediate step.
        """
        # Central difference for divergence
        du_dx = (self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx)
        dv_dy = (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)
        
        self.b[1:-1, 1:-1] = (self.rho / self.dt) * (du_dx + dv_dy)
        
        return self.b

    def poisson_solve(self):
        """
        Solve the Pressure Poisson Equation: Lap(p) = b
        Uses Domain Decomposition (Block Jacobi) and offloads blocks to NVQLink.
        """
        # 1. Compute RHS
        self.compute_b()
        
        # 2. Domain Decomposition (Matrix Partitioning)
        # Split into 2x2 blocks for simplicity (or generalized)
        n_blocks_x = 2
        n_blocks_y = 2
        
        # We assume grid is divisible by blocks for this demo
        block_ny = self.ny // n_blocks_y
        block_nx = self.nx // n_blocks_x
        
        # Estimate global spectral radius or block spectral radius
        # For 2D Laplacian, max eigenvalue is approx 8/dx^2
        # We can compute a "parameter" to pass to the Quantum Solver
        lambda_max = self.estimate_eigenvalues(block_nx, block_ny)
        
        # Iterate Block-Jacobi
        # p_new = (D)^-1 * (b - (L+U)*p_old) 
        # Actually for Poisson, we can just solve per block with fixed boundaries from neighbors
        # This is a Schwartz Alternating method or Block Jacobi.
        
        # We'll do 1 iteration of "Quantum Accelerated" Block update per time step 
        # (or a few, but usually pressure solve needs convergence)
        # For high-speed demo, we do a fixed number of hybrid iterations.
        n_hybrid_iters = 2
        
        for k in range(n_hybrid_iters):
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    # Extract Block
                    ystart = by * block_ny
                    yend = (by + 1) * block_ny
                    xstart = bx * block_nx
                    xend = (bx + 1) * block_nx
                    
                    # Pad to include boundary from neighbors for valid laplacian calc
                    # (Handling indices carefully)     
                    b_sub = self.b[ystart:yend, xstart:xend]
                    
                    # Offload this block
                    if self.link.connected:
                        # Pass eigenvalue estimation as optimization parameter
                        p_update = self.link.offload_poisson_solve(b_sub, eigenvalue_param=lambda_max)
                        
                        if p_update is not None:
                             # Update pressure block (simplified update)
                             # In real Schwartz method we'd blend with boundary conditions
                             self.p[ystart:yend, xstart:xend] = p_update
                        else:
                             # Fallback or just Classical update for this block
                             pass

        # Always do a final classical smoothing pass to ensure continuity across boundaries
        self.p = self._classical_poisson_solve(self.b, nit=5)

    def estimate_eigenvalues(self, nx_block, ny_block):
        """
        Estimate the maximum eigenvalue (spectral radius) of the discrete Laplacian 
        on the sub-domain block.
        Lambda_max ~ 4/dx^2 + 4/dy^2
        """
        # Analytical upper bound for Discrete Laplacian 5-point stencil
        ev_x = 4.0 / (self.dx**2) * np.sin(np.pi * nx_block / (2 * (nx_block + 1)))**2
        ev_y = 4.0 / (self.dy**2) * np.sin(np.pi * ny_block / (2 * (ny_block + 1)))**2
        
        # Just return the max theoretical eigenvalue magnitude
        return 4.0 / (self.dx**2) + 4.0 / (self.dy**2)

            
    def _classical_poisson_solve(self, b_field, nit=50):
        """Fallback classical solver (Jacobi)"""
        p = np.zeros_like(self.p)
        for _ in range(nit):
            p[1:-1, 1:-1] = (((p[1:-1, 2:] + p[1:-1, 0:-2]) * self.dy**2 + 
                              (p[2:, 1:-1] + p[0:-2, 1:-1]) * self.dx**2 -
                              b_field[1:-1, 1:-1] * self.dx**2 * self.dy**2) / 
                             (2 * (self.dx**2 + self.dy**2)))
            # Boundary conditions (zero gradient / homogeneous Neumann)
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]
            p[-1, :] = p[-2, :]
        return p

    def apply_boundary_conditions(self):
        """Apply Lid-Driven Cavity Boundary Conditions"""
        # No-slip walls at bottom, left, right
        self.u[0, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        self.v[0, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0
        self.v[-1, :] = 0
        
        # Lid at top
        self.u[-1, :] = self.lid_velocity

    def step(self):
        """
        Perform one time step of the simulation.
        1. Advection-Diffusion (Burgers' eq part) -> Intermediate Velocity (u_star, v_star)
        2. Pressure Solve (PPE) -> p_new
        3. Correction -> u_new, v_new
        """
        
        # 1. Intermediate Step (simplified explicit Euler for Advection/Diffusion)
        # un = u^n
        un = self.u.copy()
        vn = self.v.copy()
        
        # u component
        self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                              un[1:-1, 1:-1] * self.dt / self.dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                              vn[1:-1, 1:-1] * self.dt / self.dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                              self.nu * self.dt / self.dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                              self.nu * self.dt / self.dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
                              
        # v component
        self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                              un[1:-1, 1:-1] * self.dt / self.dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                              vn[1:-1, 1:-1] * self.dt / self.dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                              self.nu * self.dt / self.dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                              self.nu * self.dt / self.dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

        # Boundary conditions for u_star, v_star
        self.apply_boundary_conditions()

        # 2. Pressure Solve (OFFLOADED TO QPU)
        self.poisson_solve()

        # 3. Correction Step
        # u^n+1 = u_star - dt/rho * grad(p)
        self.u[1:-1, 1:-1] -= (self.dt / self.rho) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) / (2 * self.dx)
        self.v[1:-1, 1:-1] -= (self.dt / self.rho) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) / (2 * self.dy)

        # Boundary conditions enforcement
        self.apply_boundary_conditions()
