from __future__ import annotations

from typing import List
import numpy as np
from tensornetwork import ncon
from numpy import linalg as LA

from src.MPOClass import MPO

class MPS:
    """
    Class for Matrix Product States (MPS), a representation of quantum many-body states.

    Attributes:
        tensors (List[np.ndarray]): Tensors defining the MPS.
        num_sites (int): Number of sites in the MPS.
        phys_dim (int): Physical dimension of the MPS.
        bond_dims (List[int]): List of the bond dimensions in the MPS.
        max_bond_dim (int): Maximum bond dimension of the MPS.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, tensors: List[np.ndarray]):
        """
        Initialize an MPS object using a list of tensors.

        Args:
            tensors (List): List of tensors defining the MPS, each of shape (bond_dim_left, phys_dim, bond_dim_right).
        """
        self.tensors = tensors

        self.num_sites = len(tensors)
        self.phys_dim = tensors[0].shape[1] 
        # self.bond_dims, self.max_bond_dim = self._calculate_bond_dimensions()
        self._check_bond_dimension_consistency()
    
    def _calculate_bond_dimensions(self) -> tuple[List[int], int]:
        """
        Calculate the bond dimensions and return the maximum bond dimension.

        Returns:
            tuple[List[int], int]: List of bond dimensions and the maximum one.
        """
        bond_dims = []
        for i in range(self.num_sites - 1):
            bond_dims.append(self.tensors[i].shape[2])
        max_bond_dim = max(bond_dims)        
        return bond_dims, max_bond_dim
    
    def _check_bond_dimension_consistency(self):
        """
        Check that the bond dimensions between adjacent tensors are consistent.
        """
        for i in range(self.num_sites - 1):
            bond_dim_right = self.tensors[i].shape[2]
            bond_dim_left = self.tensors[i + 1].shape[0]
            if bond_dim_right != bond_dim_left:
                raise ValueError(f"Bond dimension mismatch between site {i} and site {i+1}: "
                                 f"bond_dim_right({bond_dim_right}) != bond_dim_left({bond_dim_left}).")
            
    # ----------------------------- #
    # 2. PUBLIC METHODS
    # ----------------------------- #

    def get_tensor(self, site: int) -> np.ndarray:
        """
        Returns the tensor at a specific site.

        Args:
            site (int): Index of the site (0-indexed).

        Returns:
            np.ndarray: Tensor at site i.
        """
        if site < 0 or site >= self.num_sites:
            raise ValueError("Site index out of range.")
        return self.tensors[site]    
            
    def set_tensor(self, site: int, new_tensor: np.ndarray):
        """
        Updates the tensor at a specific site.

        Args:
            site (int): Index of the site (0-indexed).
            new_tensor (np.ndarray): New tensor to be set.
        """
        if site < 0 or site >= self.num_sites:
            raise ValueError("Site index out of range.")        
        self.tensors[site] = new_tensor        
        self.bond_dims, self.max_bond_dim = self._calculate_bond_dimensions()
    
    def norm(self) -> float:
        """
        Compute the norm of the MPS by contracting it with its conjugate.

        Return:
            float: The computed norm.
        """
        tensor = self.get_tensor(0)
        norm_squared = ncon([tensor, np.conj(tensor)], [[-1, 1, -2], [-3, 1, -4]])
        for i in range(1, self.num_sites):
            tensor = self.get_tensor(i)
            norm_squared = ncon([norm_squared, tensor, np.conj(tensor)], [[-1, 1, -3, 3], [1, 2, -2], [3, 2, -4]])        
        return np.sqrt(norm_squared.item())
    
    def flat_trace(self) -> float:
        from basic_operators import flat
        tensor = self.get_tensor(0)
        trace = ncon([tensor, flat.reshape(2)], [[-1, 1, -2], [1]])
        for i in range(1, self.num_sites):
            tensor = self.get_tensor(i)
            trace = ncon([trace, tensor, flat.reshape(2)], [[-1, 1], [1, 2, -2], [2]])       
        return trace.item()
    
    def overlap(self, mps: MPS) -> float:
        """
        Compute the inner product <mps1 | mps2> between two MPS objects.

        Args:
            mps (MPS): The MPS for which the overlap is to be computed.

        Returns:
            float: The computed overlap.
        """
        if self.num_sites != mps.num_sites:
            raise ValueError("MPSs must have the same number of sites")        
        overlap = ncon([self.get_tensor(0), np.conj(mps.get_tensor(0))], [[-1, 1, -2], [-3, 1, -4]])
        for i in range(1, self.num_sites):
            overlap = ncon([overlap, self.get_tensor(i), np.conj(mps.get_tensor(i))], [[-1, 1, -3, 3], [1, 2, -2], [3, 2, -4]])
        return overlap.item()
    
    def apply_mpo(self, mpo: MPO) -> MPS:
        """
        Apply an MPO operator to the MPS and return the resulting MPS.

        Args:
            mpo (MPO): The MPO to be applied.

        Returns:
            MPS: The new MPS after applying the MPO.
        """
        if self.num_sites != mpo.num_sites:
            raise ValueError("MPS and MPO must have the same number of sites")
        result_tensors = []
        for i in range(self.num_sites):
            result_tensor = ncon([self.get_tensor(i), mpo.get_tensor(i)], [[-1, 1, -4], [-2, -5, -3, 1]])
            result_tensor = result_tensor.reshape(result_tensor.shape[0] * result_tensor.shape[1], result_tensor.shape[2], result_tensor.shape[3] * result_tensor.shape[4])
            result_tensors.append(result_tensor)
        return MPS(result_tensors)
    
    def compute_exp_val(self, mpo: MPO) -> float:
        if self.num_sites != mpo.num_sites:
            raise ValueError("MPS and MPO must have the same number of sites")        
        value = ncon([self.get_tensor(0), mpo.get_tensor(0), self.get_tensor(0).conj()], [[-1, 1, -4], [-2, -5, 2, 1], [-3, 2, -6]])
        for i in range(1, self.num_sites):
            value = ncon([value, self.get_tensor(i), mpo.get_tensor(i), self.get_tensor(i).conj()], [[-1, -2, -3, 1, 3, 5], [1, 2, -4], [3, -5, 4, 2], [5, 4, -6]])
        return value.item()

    def normalize(self, return_norm: bool = False) -> MPS | tuple[MPS, float]:
        """
        Normalize the MPS.

        Args: 
            return_norm (bool, optional): If True, return the norm of the original MPS.

        Returns:
            MPS: The normalized MPS.
            tuple[MPS, float]: If return_norm is True, returns a tuple containing the normalized MPS 
                            and the original norm.
        """
        norm = self.norm()
        if np.isclose(norm, 0):
            raise ValueError("Cannot normalize MPS: computed norm is zero.")
        normalized_tensors = [tensor / norm**(1/self.num_sites) for tensor in self.tensors]
        normalized_mps = MPS(normalized_tensors)
        if return_norm:
            return normalized_mps, norm
        return normalized_mps
    
    def truncate(self, trunc_threshold: float, max_bond_dim: int) -> MPS:
        """
        Perform SVD-based truncation on the MPS to reduce bond dimensions while controlling truncation error.

        Args:
            trunc_threshold (float): Maximum allowed truncation error.
            max_bond_dim: Maximum number of singular values to retain.

        Returns:
            MPS: The truncated MPS.
        """
        result_tensors = [self.get_tensor(i) for i in range(self.num_sites)]
        # Left-to-right SVD sweep
        for n in range(self.num_sites - 1):
            tensor = result_tensors[n]
            chi_l, d, chi_r = tensor.shape
            U, S, V = LA.svd(tensor.reshape(chi_l * d ,chi_r), full_matrices=False)
            result_tensors[n] = U.reshape(chi_l, d, U.shape[1])
            result_tensors[n + 1] = ncon([np.diag(S) @ V, result_tensors[n + 1]], [[-1, 1], [1, -2, -3]])
        # Right-to-left SVD sweep
        for n in range(self.num_sites - 1, 0, -1):
            tensor = result_tensors[n]
            chi_l, d, chi_r = tensor.shape
            U, S, V = LA.svd(tensor.reshape(chi_l, d * chi_r), full_matrices=False)
            chi_opt = self._compute_optimal_chi(S, trunc_threshold, max_bond_dim)     
            S = np.diag(S[:chi_opt] / np.maximum(LA.norm(S[:chi_opt]), 1e-12))
            result_tensors[n] = V[:chi_opt,:].reshape(chi_opt, d, chi_r)
            result_tensors[n - 1] = ncon([result_tensors[n - 1], U[:,:chi_opt] @ S], [[-1, -2, 1], [1, -3]])        
        return MPS(result_tensors)
    
    def contract(self) -> np.ndarray:
        contraction = self.get_tensor(0)
        for i in range(1, self.num_sites):
            tensor = self.get_tensor(i)
            contraction = ncon([contraction, tensor], [[-1, -2, 1], [1, -3, -4]])
            contraction = contraction.reshape(contraction.shape[0], contraction.shape[1] * contraction.shape[2], contraction.shape[-1])
        return contraction

    # ----------------------------- #
    # 3. PRIVATE METHODS
    # ----------------------------- #

    def _compute_optimal_chi(self, S: List, trunc_threshold: float, max_bond_dim: int) -> int:
        """
        Compute the optimal number of singular values to keep.

        Args:
            S (List): List of singular values.
            trunc_threshold: Maximum allowed truncation error.
            max_bond_dim: Maximum number of singular values to retain

        Returns:
            int: Optimal bond dimension value.
        """
        norm_S_squared = LA.norm(S) ** 2
        error = 0
        for s, value in enumerate(reversed(S), start=1):
            error += (value ** 2) / norm_S_squared
            if error > trunc_threshold:
                return min(len(S) - s + 1, max_bond_dim)
        return min(len(S), max_bond_dim)
    
    # ----------------------------- #
    # 4. SPECIAL METHODS
    # ----------------------------- #
    
    def __str__(self) -> str:
        """
        String representation of the MPS.
        """
        return f"MPS(num_sites={self.num_sites}, d={self.phys_dim}, bond_dims={self.bond_dims}, max_bond_dim={self.max_bond_dim})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the MPS.
        """
        return self.__str__()