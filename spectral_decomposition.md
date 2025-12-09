# Spectral Decomposition and Matrix Partitioning in Quantum CFD

## 1. Discrete Laplacian and Eigenvalues
For a 2D domain $\Omega$ discretized on a grid of size $N_x \times N_y$ with spacing $\Delta x, \Delta y$, the discrete Laplacian operator $L$ (using a 5-point stencil) has the following eigenvalues:

$$
\lambda_{k,l} = \frac{4}{\Delta x^2} \sin^2\left(\frac{k \pi}{2(N_x+1)}\right) + \frac{4}{\Delta y^2} \sin^2\left(\frac{l \pi}{2(N_y+1)}\right)
$$

where $k=1,\dots,N_x$ and $l=1,\dots,N_y$.

### Spectral Radius
The spectral radius $\rho(A)$ determines the convergence rate of iterative solvers like Jacobi or Richardson integration. It is given by the maximum eigenvalue intensity:

$$
\lambda_{\max} \approx \frac{4}{\Delta x^2} + \frac{4}{\Delta y^2}
$$

In our solver, we calculate this analytically for each partitioned block to optimize the "quantum" offload correction step.

## 2. Domain Decomposition (Schwartz Alternating Method)
To solve the linear system $Ax=b$ on a quantum-classical hybrid, we partition $\Omega$ into overlapping subdomains $\Omega_1, \Omega_2, \dots \Omega_K$.

For two domains $\Omega_1, \Omega_2$:
1. Solve $L u_1^{n+1} = f$ on $\Omega_1$ using boundary conditions from $u_2^n$.
2. Solve $L u_2^{n+1} = f$ on $\Omega_2$ using boundary conditions from $u_1^{n+1}$.

We implement a **Block-Jacobi** variant where blocks are solved in parallel (simulating parallel QPU calls), which requires damping or overlap to converge.

## 3. Direct Numerical Simulation (DNS) Observers
To physically validate that our "Quantum Statistical" initialization produces turbulence-like features, we calculate the Kinetic Energy Spectrum $E(k)$.

The Kinetic Energy is defined as:
$$
E = \frac{1}{2} \int \mathbf{u} \cdot \mathbf{u} \, dV
$$

In spectral space (Fourier domain):
$$
\hat{\mathbf{u}}(\mathbf{k}) = \text{FFT}(\mathbf{u}(\mathbf{x}))
$$
The energy spectrum $E(k)$ represents energy contained in eddies of wavenumber $k = |\mathbf{k}|$. For Kolmogorov turbulence, we expect:
$$
E(k) \propto k^{-5/3}
$$
Creating this observer allows us to quantify the flow regime.
