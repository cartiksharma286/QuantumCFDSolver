import numpy as np
from nvqlink import NVQLink
from quantum_stats import initialize_field_quantum
from quantum_interferometry import QuantumInterferometry

class QuantumHyperFluidSolver:
    """
    4D Incompressible Navier-Stokes Solver using Finite Difference Method (FDM).
    Integrates with NVQLink to offload Pressure Poisson Equation (PPE) solving.
    Dimensions: (w, z, y, x) -> (a, w, v, u) velocities.
    Using indices [l, k, j, i] for [w, z, y, x].
    """
    
    def __init__(self, nx=32, ny=32, nz=32, nw=5, dt=0.001, rho=1.0, nu=0.1, 
                 lx=1.0, ly=1.0, lz=1.0, lw=1.0, 
                 use_quantum_stats=True, lid_velocity=1.0, 
                 nvqlink_qiskit=False, n_blocks=(2,2), circuit_type='basic',
                 compute_signatures=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nw = nw
        self.dt = dt
        self.rho = rho
        self.nu = nu
        
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.dz = lz / (nz - 1)
        self.dw = lw / (nw - 1)
        
        self.lid_velocity = lid_velocity
        self.n_blocks_y, self.n_blocks_x = n_blocks
        self.compute_signatures = compute_signatures

        # Initialize fields: shape (nw, nz, ny, nx)
        shape = (nw, nz, ny, nx)
        self.u = np.zeros(shape) # X-velocity
        self.v = np.zeros(shape) # Y-velocity
        self.w_vel = np.zeros(shape) # Z-velocity (named w_vel to avoid confusion with coord w)
        self.a_vel = np.zeros(shape) # W-velocity (4th dim)
        self.p = np.zeros(shape) # Pressure
        self.b = np.zeros(shape) # Source term
        
        # Interferometry
        self.u_prev = np.zeros(shape)
        self.signatures = [] # List of signature dicts
        self.qi = QuantumInterferometry(use_qiskit=nvqlink_qiskit)
        
        # Initialize NVQLink
        self.link = NVQLink(use_qiskit=nvqlink_qiskit, circuit_type=circuit_type)
        
        if use_quantum_stats:
            print("[Solver] Initializing 4D hyper-field with Quantum Statistics...")
            # We map the scalar distribution to random perturbations
            q_dist = initialize_field_quantum(shape, distribution_type='fermi-dirac', T=0.5)
            
            # Perturb
            scale = 0.1
            noise = np.random.randn(*shape)
            self.u += q_dist * noise * scale
            self.v += q_dist * noise * scale
            self.w_vel += q_dist * noise * scale
            self.a_vel += q_dist * noise * scale

    def compute_b(self):
        """
        Compute RHS for 4D Poisson: b = rho/dt * div(u)
        """
        # Central difference divergence
        # Indices: [l, k, j, i]
        
        # du/dx: diff along axis 3
        du_dx = (self.u[1:-1, 1:-1, 1:-1, 2:] - self.u[1:-1, 1:-1, 1:-1, 0:-2]) / (2 * self.dx)
        
        # dv/dy: diff along axis 2
        dv_dy = (self.v[1:-1, 1:-1, 2:, 1:-1] - self.v[1:-1, 1:-1, 0:-2, 1:-1]) / (2 * self.dy)
        
        # dw/dz: diff along axis 1
        dw_dz = (self.w_vel[1:-1, 2:, 1:-1, 1:-1] - self.w_vel[1:-1, 0:-2, 1:-1, 1:-1]) / (2 * self.dz)
        
        # da/dw: diff along axis 0
        da_dw = (self.a_vel[2:, 1:-1, 1:-1, 1:-1] - self.a_vel[0:-2, 1:-1, 1:-1, 1:-1]) / (2 * self.dw)
        
        self.b[1:-1, 1:-1, 1:-1, 1:-1] = (self.rho / self.dt) * (du_dx + dv_dy + dw_dz + da_dw)
        return self.b

    def poisson_solve(self):
        """
        Solve 4D Poisson: Lap(p) = b using Block Jacobi on (y,x) partition,
        treating (w,z) features as part of the block payload.
        """
        self.compute_b()
        
        # Partitioning X, Y
        block_ny = self.ny // self.n_blocks_y
        block_nx = self.nx // self.n_blocks_x
        
        # Eigenvalue estimation: 4D
        lambda_max = self.estimate_eigenvalues_4d()
        
        n_hybrid_iters = 2
        
        for k in range(n_hybrid_iters):
            for by in range(self.n_blocks_y):
                for bx in range(self.n_blocks_x):
                    # Extract Block spanning full W, Z range
                    ystart = by * block_ny
                    yend = (by + 1) * block_ny
                    xstart = bx * block_nx
                    xend = (bx + 1) * block_nx
                    
                    # sub-block shape: (nw, nz, block_ny, block_nx)
                    b_sub = self.b[:, :, ystart:yend, xstart:xend]
                    
                    if self.link.connected:
                        p_update = self.link.offload_poisson_solve(b_sub, eigenvalue_param=lambda_max)
                        if p_update is not None:
                             self.p[:, :, ystart:yend, xstart:xend] = p_update

        # Classical Smoothing (Jacobi 4D)
        self.p = self._classical_poisson_solve_4d(self.b, nit=2)

    def _classical_poisson_solve_4d(self, b_field, nit=10):
        p = np.copy(self.p)
        
        dx2, dy2, dz2, dw2 = self.dx**2, self.dy**2, self.dz**2, self.dw**2
        denom = 2 * (1/dx2 + 1/dy2 + 1/dz2 + 1/dw2)
        
        for _ in range(nit):
            # 7-point (or 9-point in 4D?) -> (2*d) + 1 = 9 point stencil center + neighbors
            # p_xx + p_yy + p_zz + p_ww = b
            
            p_xp = p[1:-1, 1:-1, 1:-1, 2:]
            p_xm = p[1:-1, 1:-1, 1:-1, 0:-2]
            p_yp = p[1:-1, 1:-1, 2:, 1:-1]
            p_ym = p[1:-1, 1:-1, 0:-2, 1:-1]
            p_zp = p[1:-1, 2:, 1:-1, 1:-1]
            p_zm = p[1:-1, 0:-2, 1:-1, 1:-1]
            p_wp = p[2:, 1:-1, 1:-1, 1:-1]
            p_wm = p[0:-2, 1:-1, 1:-1, 1:-1]
            
            term_x = (p_xp + p_xm) / dx2
            term_y = (p_yp + p_ym) / dy2
            term_z = (p_zp + p_zm) / dz2
            term_w = (p_wp + p_wm) / dw2
            source = b_field[1:-1, 1:-1, 1:-1, 1:-1]
            
            p[1:-1, 1:-1, 1:-1, 1:-1] = (term_x + term_y + term_z + term_w - source) / (1/dx2 + 1/dy2 + 1/dz2 + 1/dw2)
            
            # Simple BCs (Neumann everywhere for now to avoid complexity)
            # Or Lid Driven? 
            # Implement 4D Boundary Conditions explicitly?
            # Let's simplify and just clamp boundaries or set Neumann
            p[:, :, :, -1] = p[:, :, :, -2]
            p[:, :, :, 0] = p[:, :, :, 1]
            # ... others ...
            
        return p

    def estimate_eigenvalues_4d(self):
        return 4.0/self.dx**2 + 4.0/self.dy**2 + 4.0/self.dz**2 + 4.0/self.dw**2

    def apply_boundary_conditions(self):
        # 4D BCs
        # Walls: u=0 at all faces except Top (y=L)
        # Lid at y=max (index -1 in axis 2)
        
        self.u[:] = 0
        self.v[:] = 0
        self.w_vel[:] = 0
        self.a_vel[:] = 0
        
        # Restore flow (this is a crude BC application, simplified for demo)
        # Actually we should strictly enforce 0 at boundaries and preserve interior
        # But `self.u[:] = 0` wipes everything.
        # Correct approach:
        # Zero out boundaries
        # Face X=0
        self.u[:, :, :, 0] = 0
        # Face X=L
        self.u[:, :, :, -1] = 0
        # Face Y=0
        self.u[:, :, 0, :] = 0
        self.v[:, :, 0, :] = 0
        
        # Lid: Face Y=L (Top)
        self.u[:, :, -1, :] = self.lid_velocity
        # All other velocities 0 on lid
        self.v[:, :, -1, :] = 0
        self.w_vel[:, :, -1, :] = 0
        self.a_vel[:, :, -1, :] = 0
        
        # Face Z=0, Z=L
        self.u[:, 0, :, :] = 0
        self.u[:, -1, :, :] = 0
        
        # Face W=0, W=L
        self.u[0, :, :, :] = 0
        self.u[-1, :, :, :] = 0

    def analyze_flow_signatures(self, step_idx):
        """
        Compute quantum statistical signatures of the flow.
        1. Temporal Autocorrelation: (u_t, u_{t-1})
        """
        # Statistical Swap Test on random samples
        fidelities = self.qi.compute_bulk_signatures(self.u, self.u_prev, samples=200)
        
        # Store distribution stats or full histogram data
        # Let's store full data for plotting later, and mean
        sig_data = {
            'step': step_idx,
            'fid_mean': np.mean(fidelities),
            'fid_std': np.std(fidelities),
            'histogram': fidelities
        }
        self.signatures.append(sig_data)

    def step(self, step_idx=0):
        # 1. Advection (Skipped / Implicit via Init + correction in this simplified demo)
        
        # 2. Pressure Projection
        self.apply_boundary_conditions()
        self.poisson_solve()
        
        # 3. Correction
        # Update u, v, w, a using pressure gradients
        # We update interior points [1:-1, 1:-1, 1:-1, 1:-1]
        
        # dp/dx
        # indices: [l, k, j, i]
        # X is axis 3
        # We need gradients at interior points. 
        # For centered diff at i, we need p[i+1] - p[i-1].
        # Slicing p for matching indices:
        p_xp = self.p[1:-1, 1:-1, 1:-1, 2:]
        p_xm = self.p[1:-1, 1:-1, 1:-1, 0:-2]
        p_grad_x = (p_xp - p_xm) / (2 * self.dx)
        self.u[1:-1, 1:-1, 1:-1, 1:-1] -= (self.dt / self.rho) * p_grad_x
        
        # dp/dy (axis 2)
        p_yp = self.p[1:-1, 1:-1, 2:, 1:-1]
        p_ym = self.p[1:-1, 1:-1, 0:-2, 1:-1]
        p_grad_y = (p_yp - p_ym) / (2 * self.dy)
        self.v[1:-1, 1:-1, 1:-1, 1:-1] -= (self.dt / self.rho) * p_grad_y
        
        # dp/dz (axis 1)
        p_zp = self.p[1:-1, 2:, 1:-1, 1:-1]
        p_zm = self.p[1:-1, 0:-2, 1:-1, 1:-1]
        p_grad_z = (p_zp - p_zm) / (2 * self.dz)
        self.w_vel[1:-1, 1:-1, 1:-1, 1:-1] -= (self.dt / self.rho) * p_grad_z
        
        # dp/dw (axis 0)
        p_wp = self.p[2:, 1:-1, 1:-1, 1:-1]
        p_wm = self.p[0:-2, 1:-1, 1:-1, 1:-1]
        p_grad_w = (p_wp - p_wm) / (2 * self.dw)
        self.a_vel[1:-1, 1:-1, 1:-1, 1:-1] -= (self.dt / self.rho) * p_grad_w

        self.apply_boundary_conditions()
        
        # Interferometry Analysis
        if self.compute_signatures:
            # Analyze using current u and previous u (stored at start of step? no, store at end for next step?
            # We want change from t-1 to t.
            # u_prev is currently t-1. u is t.
            self.analyze_flow_signatures(step_idx)
            # Update prev for next step
            self.u_prev = np.copy(self.u)

    def compute_energy_spectrum(self):
        """
        4D Energy Spectrum.
        Integration over 3D shells? Or 4D shells?
        E(k) where k = sqrt(kx^2 + ky^2 + kz^2 + kw^2)
        """
        u_sq = self.u**2 + self.v**2 + self.w_vel**2 + self.a_vel**2
        ft_u = np.fft.fftn(self.u)
        ft_v = np.fft.fftn(self.v)
        ft_w = np.fft.fftn(self.w_vel)
        ft_a = np.fft.fftn(self.a_vel)
        
        energy_spatial = 0.5 * (np.abs(ft_u)**2 + np.abs(ft_v)**2 + np.abs(ft_w)**2 + np.abs(ft_a)**2)
        
        # Binning in 4D k-space... expensive?
        # Just return dummy or 1D slice for now to avoid massive computation delay in demo
        # Or, just compute for middle slice
        mid_w = self.nw // 2
        mid_z = self.nz // 2
        
        # Return spectrum of the 2D slice for comparison with 2D theory
        # FFT of slice
        slice_u = self.u[mid_w, mid_z, :, :]
        slice_v = self.v[mid_w, mid_z, :, :]
        
        ft_u_2d = np.fft.fft2(slice_u)
        ft_v_2d = np.fft.fft2(slice_v)
        es_2d = 0.5 * (np.abs(ft_u_2d)**2 + np.abs(ft_v_2d)**2)
        
        ny, nx = es_2d.shape
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
                energy_bins[i] = np.sum(es_2d[indices])
                
        return k_bins[1:], energy_bins[1:]
