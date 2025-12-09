import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import EfficientSU2, TwoLocal, IQP
except ImportError:
    QuantumCircuit = None
    EfficientSU2 = None
    TwoLocal = None
    IQP = None

def build_circuit(circuit_type, num_qubits=2, param_val=0.0, reps=1):
    """
    Factory function to build a Qiskit quantum circuit based on the type.
    
    Args:
        circuit_type (str): Type of circuit ('basic', 'efficient_su2', 'amplitude_encoding', 'angle_encoding', 'iqp').
        num_qubits (int): Number of qubits.
        param_val (float): A parameter to encode into the circuit (e.g. rotation angle).
        reps (int): Depth/Repetitions of the variational circuit.
        
    Returns:
        QuantumCircuit: The constructed Qiskit circuit, or None if Qiskit not installed.
    """
    if QuantumCircuit is None:
        return None
        
    qc = QuantumCircuit(num_qubits)
    
    if circuit_type == 'basic':
        # Simple H + RX encoding
        qc.h(range(num_qubits))
        qc.rx(param_val, 0)
        qc.cx(0, 1)
        qc.measure_all()
        
    elif circuit_type == 'efficient_su2':
        # Hardware efficient ansatz
        # Encode parameter into the first rotation
        qc.ry(param_val, 0)
        # Add EfficientSU2 block with variable depth
        ansatz = EfficientSU2(num_qubits, reps=reps)
        qc.compose(ansatz, inplace=True)
        qc.measure_all()
        
    elif circuit_type == 'amplitude_encoding':
        # Mock amplitude encoding logic
        # State = cos(theta)|0> + sin(theta)|1>
        qc.ry(param_val * 2, range(num_qubits))
        qc.measure_all()

    elif circuit_type == 'angle_encoding':
        # Qubit Angle Encoding: Ry(x) on all qubits
        # We can increase depth by repeating the encoding layer with CNOTs entanglement
        for _ in range(reps):
            qc.ry(param_val, range(num_qubits))
            if num_qubits > 1:
                # Linear entanglement
                for i in range(num_qubits - 1):
                    qc.cx(i, i+1)
        qc.measure_all()
        
    elif circuit_type == 'iqp':
        # Instantaneous Quantum Polynomial encoding
        # Hard to simulate classically. Requires IQP circuit library or manual construction.
        qc.h(range(num_qubits))
        # Z-rotations based on param
        for i in range(num_qubits):
            qc.rz(param_val, i)
        # ZZ iteractions
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                # Rzz gate
                qc.cx(i, j)
                qc.rz(param_val, j)
                qc.cx(i, j)
        
        # If IQP library available and requested (usually takes matrix interaction)
        # Here we did manual implementation for demonstrative 'reps' (repeating layers)
        if reps > 1:
            # Just repeat the unitary block
            pass 
            
        qc.h(range(num_qubits))
        qc.measure_all()
        
    else:
        # Default to basic
        print(f"[QuantumCircuits] Unknown type '{circuit_type}', using basic.")
        qc.h(range(num_qubits))
        qc.rx(param_val, 0)
        qc.measure_all()
        
    return qc
