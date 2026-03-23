import cmath
import numpy as np
from tensornetwork import ncon
from scipy.sparse.linalg import expm

from basic_operators import identity_matrix, proj_ket1, flat, ket0, ket1
from src.MPOClass import MPO


class MPOConstructors:
    """
    A class to construct Matrix Product Operators (MPOs) that are 
    useful for our model for both real and imaginary exponentials.

    Attributes:
        num_sites (int): Number of sites in the MPO.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, num_sites: int):
        """
        Initialize an MPOConstructors object for a number of sites.

        Args:
            num_sites (int): Number of sites in the MPO.
        """
        self.num_sites = num_sites

    # ----------------------------- #
    # 2. PUBLIC METHODS
    # ----------------------------- #

    def on_site_observable(self, coefficient: float, operator: np.ndarray, imaginary: bool = False) -> MPO:
        """
        Construct an MPO for an on-site observable term, either real or imaginary.

        Args:
            coefficient (float): Coefficient for the exponential.
            operator (np.ndarray): The operator for the MPO (e.g., a Pauli X matrix).
            imaginary (bool, optional): Whether the exponent is imaginary (default False).

        Returns:
            MPO: The constructed MPO.
        """
        exp_factor = -1j * coefficient if imaginary else -coefficient
        bulk = np.zeros([1, 1, 2, 2], dtype=complex)
        bulk[0, 0, :, :] = expm(exp_factor * operator)
        left_boundary = bulk
        right_boundary = bulk
        return MPO([left_boundary] + [bulk] * (self.num_sites - 2) + [right_boundary])

    def two_body_correlation(self, coefficient: float, operator: np.ndarray, imaginary: bool = False) -> MPO:
        """
        Construct an MPO for two-body correlation terms, either real or imaginary.

        Args:
            coefficient (float): Coefficient for the exponential.
            operator (np.ndarray): The operator for the MPO (e.g., a X_i X_i+1 term).
            imaginary (bool, optional): Whether the exponent is imaginary (default False).

        Returns:
            MPO: The constructed MPO.
        """
        left_boundary = np.array([1, 1])
        right_boundary = np.array([1, 1])
        bulk = np.zeros([2, 2, 2, 2], dtype=complex)
        bulk[0, 0, :, :] = identity_matrix
        exp_factor = -1j * coefficient if imaginary else -coefficient
        scaling_factor = cmath.sqrt(np.exp(exp_factor) - 1)
        bulk[1, 1, :, :] = scaling_factor * operator
        left_boundary = ncon(
            [left_boundary, bulk],
            [[1], [1, -1, -2, -3]]
        )
        right_boundary = ncon(
            [bulk, right_boundary],
            [[-1, 1, -2, -3], [1]]
        )
        bulk = ncon(
            [left_boundary, right_boundary],
            [[-2, -3, 1], [-1, 1, -4]]
        ).reshape(2, 2, 2, 2)
        return MPO([left_boundary.reshape(1, 2, 2, 2)] + [bulk] * (self.num_sites - 2) + [right_boundary.reshape(2, 1, 2, 2)])
    
    def magnetization(self, operator: np.ndarray) -> MPO:
        """
        Construct an MPO for computing a magnetization-like operator.

        Args:
            operator (np.ndarray): The operator for the magnetization (e.g., X).

        Returns:
            MPO: The constructed MPO.
        """
        left_boundary = np.array([1, 0])
        right_boundary = np.array([0, 1])
        bulk = np.zeros([2, 2, 2, 2], dtype=complex)
        bulk[0, 0, :, :] = identity_matrix
        bulk[0, 1, :, :] = operator
        print(operator)
        bulk[1, 1, :, :] = identity_matrix
        left_boundary = ncon(
            [left_boundary, bulk],
            [[1], [1, -1, -2, -3]]
        )
        right_boundary = ncon(
            [bulk, right_boundary],
            [[-1, 1, -2, -3], [1]]
        )
        return MPO([left_boundary.reshape(1, 2, 2, 2)] + [bulk] * (self.num_sites - 2) + [right_boundary.reshape(2, 1, 2, 2)])
    
