from scipy.sparse.linalg import expm
from basic_operators import pauli_x, proj_ket0

import numpy as np


class Jump:
    """
    Represents the system-ancilla interaction in our model with Trotterized dynamics.
    
    Attributes:
        gamma (float): System-ancilla interaction strength.
        Delta_t (float): Colision time step.
        delta_t (float): Trotter time step.
        interaction_tensor (np.ndarray): Tensor describing the system-ancilla interaction.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, gamma: float, Delta_t: float, delta_t: float):
        """
        Initialize a Jump instance using some parameters.

        Args:
            gamma (float): System-ancilla interaction strength.
            Delta_t (float): Colision time step.
            delta_t (float): Trotter time step.
        """
        self.gamma = gamma
        self.Delta_t = Delta_t
        self.delta_t = delta_t

        self.interaction_tensor = self._compute_interaction()

    # ----------------------------- #
    # 2. PRIVATE METHODS
    # ----------------------------- #

    def _compute_interaction(self):
        """
        Compute the tensor for the ancilla-system interaction.

        Returns:
            numpy.ndarray: Interaction tensor of shape (4, 4).
        """
        coupling = np.sqrt(self.gamma / self.Delta_t)
        interaction_op = expm(
            -1j * coupling * self.delta_t * np.kron(proj_ket0, pauli_x)
        )
        """Igual no hace falta hacer el reshape!!!"""
        return interaction_op.reshape(4, 4)
    
    # ----------------------------- #
    # 3. SPECIAL METHODS
    # ----------------------------- #

    def __str__(self) -> str:
        """
        String representation of the Jump.
        """
        return f"Jump(gamma={self.gamma}, Delta_t={self.Delta_t}, delta_t={self.delta_t})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the Jump.
        """
        return self.__str__()