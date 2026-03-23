
from __future__ import annotations

from copy import deepcopy
import logging
from typing import List
import numpy as np
from scipy.sparse.linalg import expm
from tensornetwork import ncon

from basic_operators import ket0, ket1, proj_ket1, identity_matrix, copy_tensor, pushpin, flat
from src.MPDOClass import MPDO
from src.MPOClass import MPO
from src.MPSClass import MPS
from src.MPOConstructors import MPOConstructors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TNsetup:
    """
    Class for setting up a tensor network representation to solve the problem.
    It constructs a Kraus map and extracts the largest eigenvalue using the power method.
    
    Attributes:
        num_sites (int): Number of sites in the system.
        observable (np.ndarray): Observable to be measured.
        s (float): Bias parameter affecting the evolution.
        system_evolution_op (tuple): Tuple containing the MPOs describing the system evolution.
        ancilla_interaction_op (np.ndarray): Tensor describing the system-ancilla interaction.
        trotter_operators (MPO): Trotterized evolution operators.
        bias (np.ndarray): Bias tensor applied during ancilla tracing (only if observable is '00').
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, num_sites: int, observable: np.ndarray, s: float, 
                 system_evolution_op: tuple, ancilla_interaction_op: np.ndarray):
        """
        Initializes the TNsetup class with the given parameters.

        Args:
            num_sites (int): Number of sites in the system.
            observable (np.ndarray): Observable to be computed.
            s (float): Bias parameter.
            system_evolution_op (tuple): MPOs describing the system evolution.
            ancilla_interaction_op (np.ndarray): Tensor describing the system-ancilla interaction.
        """
        self.num_sites = num_sites
        self.observable = observable
        self.s = s
        self.system_evolution_op = system_evolution_op
        self.ancilla_interaction_op = ancilla_interaction_op
        
        self.trotter_operators = self._define_trotter_operators()
        
        if self.observable == '00':
            self.bias = self._compute_activity_bias()
            self.derivative_bias = self._compute_derivative_activity_bias()

    def _define_trotter_operators(self) -> MPO:
        """
        Defines the Trotterized evolution operators for the system.

        Returns:
            MPO: Evolution operators applied during the Trotter step.
        """
        trotter_operators = []
        for i in range(self.num_sites):
            tensor_site = self.system_evolution_op[0].get_tensor(i)
            tensor_two_site = self.system_evolution_op[1].get_tensor(i)
            trotter_operators.append(ncon(
                [self.ancilla_interaction_op, tensor_site, tensor_two_site],
                [[1, -6], [-1, -3, 2, 1], [-2, -4, -5, 2]]
            ).reshape(tensor_two_site.shape))
        return MPO(trotter_operators)
    
    def _compute_activity_bias(self) -> np.ndarray:
        """
        Computes the bias tensor applied during ancilla tracing.
        
        Returns:
            np.ndarray: Bias tensor.
        """
        return expm(-self.s * proj_ket1)
    
    def _compute_derivative_activity_bias(self) -> np.ndarray:
        """
        Computes the bias tensor applied for extracting the activity.

        Returns:
            np.ndarray: Bias tensor.
        """
        magnetization = MPOConstructors(self.num_sites).magnetization(proj_ket1)
        bias = np.zeros([1, 1, 2, 2], dtype=complex)
        bias[0, 0, :, :] = self.bias
        bias_mpo = MPO([bias for i in range(self.num_sites)])
        return bias_mpo.apply_mpo(magnetization)
    
    # ----------------------------- #
    # 2. PUBLIC METHODS
    # ----------------------------- #

    def apply_collision_step(self, mixed_state: MPDO, num_trotter_steps: int, 
                             trunc_threshold: float, max_bond_dim: int) -> MPDO:
        """
        Evolves the mixed state using the Trotterized operators.
        
        Args:
            mixed_state (MPDO): Initial mixed state.
            num_trotter_steps (int): Number of Trotter steps per iteration.
            trunc_threshold (float): Threshold for truncation.
            max_bond_dim (int): Maximum bond dimension.
        
        Returns:
            MPDO: Evolved mixed state.
        """
        mixed_state = self._initialize_ancillas(mixed_state)
        for _ in range(num_trotter_steps):
            mixed_state = mixed_state.apply_map(self.trotter_operators)
            mixed_state = mixed_state.truncate(trunc_threshold, max_bond_dim)
        return mixed_state
    
    def find_steady_state(self, mixed_state: MPDO, max_iterations: int, tolerance: float, 
                          num_trotter_steps: int, trunc_threshold: float, max_bond_dim: int) -> tuple[List[float], MPDO]:
        """
        Finds the steady-state solution using the power method.
        
        Args:
            mixed_state (MPDO): Initial mixed state.
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence threshold.
            num_trotter_steps (int): Number of Trotter steps per iteration.
            trunc_threshold (float): Truncation threshold.
            max_bond_dim (int): Maximum bond dimension.
        
        Returns:
            tuple[List[float], MPDO]: Convergence history of eigenvalues and the steady-state mixed state.
        """
        eigenvalues = []
        previous_state = deepcopy(mixed_state)
        for iteration in range(0, max_iterations, 10):
            logging.info(f"Starting block {iteration} to {iteration + 10}")
            for _ in range(10):
                if len(eigenvalues) >= max_iterations:
                    break
                mixed_state = self.apply_collision_step(previous_state, num_trotter_steps, trunc_threshold, max_bond_dim)
                mixed_state = self._contract_ancilla_legs(mixed_state)
                mixed_state, norm = mixed_state.truncate(trunc_threshold, 4 ** (self.num_sites // 2), return_norm=True)
                print('Eigenvalue:', norm)
                eigenvalues.append(norm)
                previous_state = deepcopy(mixed_state)
            if len(eigenvalues) >= 10 and abs(eigenvalues[-1] - eigenvalues[-10]) / abs(eigenvalues[-10]) < tolerance:
                return eigenvalues, mixed_state
        return eigenvalues, mixed_state

    
    def find_eigenvalue_derivative(self, steady_state: MPDO, num_trotter_steps: int, trunc_threshold: float, max_bond_dim: int) -> float:
        mixed_state = self.apply_collision_step(steady_state, num_trotter_steps, trunc_threshold, max_bond_dim)
        mixed_state = self._contract_ancilla_legs_d(mixed_state)
        mixed_state, norm = mixed_state.truncate(trunc_threshold, 4 ** (self.num_sites // 2), return_norm=True)
        lambda_derivative = mixed_state.overlap(steady_state) * norm
        return lambda_derivative
    
    # ----------------------------- #
    # 3. PRIVATE METHODS
    # ----------------------------- #
    
    def _initialize_ancillas(self, mixed_state: MPDO) -> MPDO:
        """
        Initializes ancillas in the |0> state and incorporates them into the system.

        Args:
            mixed_state (MPDO): Mixed state of the system.

        Returns:
            mixed_state (MPDO): Mixed state of the system witn ancillas.
        """
        result_tensors = [mixed_state.get_tensor(i) for i in range(self.num_sites)]
        for i in range(self.num_sites):
            tensor = result_tensors[i]
            result_tensors[i] = ncon([tensor, ket0.full().reshape(2), ket0.full().reshape(2)], [[-1, -2, -4, -6], [-3], [-5]]).reshape(tensor.shape[0], 2 * mixed_state.phys_dim, 2 * mixed_state.phys_dim, tensor.shape[-1])
        return MPDO(result_tensors)
    
    def _contract_ancilla_legs(self, mixed_state: MPDO) -> None:
        """
        Contracts ancilla legs with the bias tensor.
        
        Args:
            mixed_state (MPDO): Mixed state of the system.

        Returns:
            mixed_state (MPDO): Mixed state of the system after tracing the ancillas.
        """
        result_tensors = [mixed_state.get_tensor(i) for i in range(self.num_sites)]
        for i in range(self.num_sites):
            tensor = result_tensors[i]
            tensor = tensor.reshape(tensor.shape[0], 2, 2, 2, 2, tensor.shape[-1])
            result_tensors[i] = ncon([tensor, self.bias], [[-1, -2, 1, -3, 2, -4], [2, 1]])
        return MPDO(result_tensors)
    
    def _contract_ancilla_legs_d(self, mixed_state: MPDO) -> None:
        """
        Contracts ancilla legs with the bias tensor.
        
        Args:
            mixed_state (MPDO): Mixed state of the system.

        Returns:
            mixed_state (MPDO): Mixed state of the system after tracing the ancillas.
        """
        result_tensors = [mixed_state.get_tensor(i) for i in range(self.num_sites)]
        for i in range(self.num_sites):
            tensor = result_tensors[i]
            tensor = tensor.reshape(tensor.shape[0], 2, 2, 2, 2, tensor.shape[-1])
            result_tensors[i] = ncon([tensor, self.derivative_bias.get_tensor(i)], [[-1, -3, 1, -4, 2, -5], [-2, -6, 2, 1]])
            result_tensors[i] = result_tensors[i].reshape(result_tensors[i].shape[0]*result_tensors[i].shape[1],result_tensors[i].shape[2],result_tensors[i].shape[3],result_tensors[i].shape[4]*result_tensors[i].shape[5])
        return MPDO(result_tensors)

    # ----------------------------- #
    # 4. SPECIAL METHODS
    # ----------------------------- #
    
    def __str__(self) -> str:
        """
        String representation of the TNSetup.
        """
        return f"TNsetup(num_sites={self.num_sites}, observable={self.observable}, s={self.s})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the MPS.
        """
        return self.__str__()