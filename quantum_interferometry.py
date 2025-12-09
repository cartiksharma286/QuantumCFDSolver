import numpy as np

try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.primitives import Sampler
except ImportError:
    QuantumCircuit = None
    Sampler = None

class QuantumInterferometry:
    """
    Implements Quantum Interferometry techniques, primarily the Swap Test,
    to measure the overlap (fidelity) between two quantum states.
    """
    
    def __init__(self, use_qiskit=False):
        self.use_qiskit = use_qiskit
        if self.use_qiskit and Sampler is not None:
            self.sampler = Sampler()
        else:
            self.sampler = None
            if use_qiskit:
                print("[QuantumInterferometry] Qiskit not found. Fallback to simulation.")
                self.use_qiskit = False

    def swap_test(self, angle_a, angle_b):
        """
        Perform a Swap Test to measure overlap magnitude squared: |<psi|phi>|^2
        between two states encoded by Ry(angle_a) and Ry(angle_b).
        
        State |psi> = Ry(a)|0>
        State |phi> = Ry(b)|0>
        
        Circuit:
        q0: -H-*-H-M-
               |
        q1: -[psi]-X-
               |
        q2: -[phi]-X-
        
        P(0) = 0.5 + 0.5 * |<psi|phi>|^2
        Fidelity = 2 * P(0) - 1
        
        Args:
            angle_a (float): Rotation angle for state A.
            angle_b (float): Rotation angle for state B.
            
        Returns:
            float: Fidelity (overlap squared), range [0, 1].
        """
        if not self.use_qiskit:
            # Classical simulation of Swap Test result
            # State vector A: [cos(a/2), sin(a/2)]
            # State vector B: [cos(b/2), sin(b/2)]
            # Overlap = (cos(a/2)cos(b/2) + sin(a/2)sin(b/2))^2 = cos((a-b)/2)^2
            fidelity = np.cos((angle_a - angle_b) / 2.0) ** 2
            
            # Add some statistical noise if desired, but clean is fine for math check
            return float(fidelity)

        # Qiskit Implementation
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Prepare states
        qc.ry(angle_a, 1)
        qc.ry(angle_b, 2)
        
        # Swap Test
        qc.h(0)
        qc.cswap(0, 1, 2)
        qc.h(0)
        
        qc.measure(0, 0)
        
        # Run
        job = self.sampler.run(qc)
        result = job.result()
        quasi_dists = result.quasi_dists[0]
        
        p0 = quasi_dists.get(0, 0.0)
        
        # Fidelity = 2*P(0) - 1
        fidelity = 2 * p0 - 1
        return max(0.0, min(1.0, fidelity)) # Clamp due to shot noise

    def compute_bulk_signatures(self, field_a, field_b, samples=100):
        """
        Compute statistical distribution of fidelities between two fields.
        Samples random points and runs Swap Test.
        
        Args:
            field_a (np.ndarray): First field (e.g. u at t).
            field_b (np.ndarray): Second field (e.g. u at t-1).
            samples (int): Number of Swap Tests to run.
            
        Returns:
            np.ndarray: Array of fidelity values.
        """
        # Flat view
        fa_flat = field_a.flatten()
        fb_flat = field_b.flatten()
        
        indices = np.random.choice(len(fa_flat), size=samples, replace=False)
        
        fidelities = []
        
        # Normalize fields roughly to [0, pi] for encoding
        # This is arbitrary mapping; critical for meaningful results in real app
        scale = np.pi / (np.max(np.abs(fa_flat)) + 1e-9)
        
        for idx in indices:
            val_a = fa_flat[idx]
            val_b = fb_flat[idx]
            
            angle_a = val_a * scale
            angle_b = val_b * scale
            
            fid = self.swap_test(angle_a, angle_b)
            fidelities.append(fid)
            
        return np.array(fidelities)
