from qutip import basis, sigmax, sigmay, sigmaz, sigmap, sigmam, qeye, Qobj
import numpy as np

# Base vectors (kets)
ket0 = basis(2, 0)  # |0⟩
ket1 = basis(2, 1)  # |1⟩

# Identity matrix
identity_matrix = qeye(2).full()

# Pauli matrices
pauli_x = sigmax().full()  # Pauli-X (NOT gate)
pauli_y = sigmay().full()  # Pauli-Y
pauli_z = sigmaz().full()  # Pauli-Z

# Ladder operators
sigma_plus = sigmap().full()  # σ+
sigma_minus = sigmam().full()  # σ-

# Projection operators
proj_ket0 = (ket0 * ket0.dag()).full()  # Projector onto |0⟩
proj_ket1 = (ket1 * ket1.dag()).full()  # Projector onto |1⟩

# Copy tensor
copy_tensor = np.zeros((2, 2, 2), dtype=int)
for i in range(2):
    copy_tensor[i, i, i] = 1

# Pushpin
pushpin = Qobj([1.0, 0.0, 0.0, 1.0]).full()
flat = Qobj([1.0, 1.0]).full().reshape(2)