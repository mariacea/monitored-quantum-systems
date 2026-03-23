
from __future__ import annotations

from typing import List
import numpy as np
from tensornetwork import ncon
from numpy import linalg as LA

from src.MPOClass import MPO

class MPDO:
    """
    Class for Matrix Product Density Operators (MPDO), a representation of quantum many-body density matrices.

    Attributes:
        tensors (List[np.ndarray]): Tensors defining the MPDO.
        num_sites (int): Number of sites in the MPDO.
        phys_dim (int): Physical dimension of the MPDO.
        bond_dims (List[int]): List of the bond dimensions in the MPDO.
        max_bond_dim (int): Maximum bond dimension of the MPDO.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, tensors: List[np.ndarray]):
        """
        Initialize an MPDO object using a list of tensors.

        Args:
            tensors (List): List of tensors defining the MPDO, each of shape (bond_dim_left, phys_dim, phys_dim, bond_dim_right).
        """
        self.tensors = tensors

        self.num_sites = len(tensors)
        self.phys_dim = self.tensors[0].shape[1] if len(self.tensors[0].shape) == 4 or len(self.tensors[0].shape) == 5 else int(np.sqrt(self.tensors[0].shape[1]))
        self.bond_dims, self.max_bond_dim = self._calculate_bond_dimensions()
        self._check_bond_dimension_consistency()
    
    def _calculate_bond_dimensions(self) -> tuple[List[int], int]:
        """
        Calculate the bond dimensions and return the maximum bond dimension.

        Returns:
            tuple[List[int], int]: List of bond dimensions and the maximum one.
        """
        bond_dims = []
        for i in range(self.num_sites - 1):
            bond_dims.append(self.tensors[i].shape[-1])
        max_bond_dim = max(bond_dims)        
        return bond_dims, max_bond_dim
    
    def _check_bond_dimension_consistency(self):
        """
        Check that the bond dimensions between adjacent tensors are consistent.
        """
        for i in range(self.num_sites - 1):
            bond_dim_right = self.tensors[i].shape[-1]
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
        norm_squared = ncon([tensor, np.conj(tensor)], [[-1, 2, 1, -3], [-2, 2, 1, -4]])
        for i in range(1, self.num_sites):
            tensor = self.get_tensor(i)
            norm_squared = ncon([norm_squared, tensor, np.conj(tensor)], [[-1, -2, 1, 2], [1, 3, 4, -3], [2, 3, 4, -4]])     
        return np.sqrt(norm_squared.item())
    
    # NEW!
    def trace_norm(self) -> float:
        from basic_operators import identity_matrix
        tensor = self.get_tensor(0)
        trace = ncon([identity_matrix, identity_matrix], [[-1, -3], [-2, -4]]).reshape(4, 4)
        norm = ncon([tensor, trace], [[-1, 1, 2, -2], [1, 2]])
        for i in range(1, self.num_sites):
            tensor = self.get_tensor(i)
            norm = ncon([norm, tensor, trace], [[-1, 1], [1, 2, 3, -2], [2, 3]])
        return norm.item()

    
    def overlap(self, mpdo: MPDO) -> float:
        """
        Compute the inner product <mps1 | mps2> between two MPS objects.

        Args:
            mpdo (MPDO): The MPDO for which the overlap is to be computed.

        Returns:
            float: The computed overlap.
        """
        if self.num_sites != mpdo.num_sites:
            raise ValueError("MPDOs must have the same number of sites") 
        overlap = ncon([self.get_tensor(0), np.conj(mpdo.get_tensor(0))], [[-1, 2, 1, -3], [-2, 2, 1, -4]])
        for i in range(1, self.num_sites):
            overlap = ncon([overlap, self.get_tensor(i), np.conj(mpdo.get_tensor(i))], [[-1, -2, 1, 2], [1, 3, 4, -3], [2, 3, 4, -4]])
        return overlap.item()
    
    def apply_map(self, mpo: MPO) -> MPDO:
        """
        Apply a map to the MPDO and return the resulting MPDO.

        Args:
            mpo (MPO): The MPO to be applied.

        Returns:
            MPDO: The new MPDO after applying the MPO.
        """
        if self.num_sites != mpo.num_sites:
            raise ValueError("MPDO and MPO must have the same number of sites")
        result_tensors = []
        for i in range(self.num_sites):
            mpo_tensor = mpo.get_tensor(i)
            result_tensor = ncon([self.get_tensor(i), mpo_tensor, np.conj(mpo_tensor)], [[-1, 1, 2, -6], [-2, -7, -4, 1], [-3, -8, -5, 2]])
            result_tensors.append(result_tensor.reshape(np.prod(result_tensor.shape[:3]).item(), mpo_tensor.shape[2], mpo_tensor.shape[2], np.prod(result_tensor.shape[-3:]).item()))
        return MPDO(result_tensors)

    def normalize(self, return_norm: bool = False) -> MPDO | tuple[MPDO, float]:
        """
        Normalize the MPDO.

        Args: 
            return_norm (bool, optional): If True, return the norm of the original MPDO.

        Returns:
            MPDO: The normalized MPDO.
            tuple[MPDO, float]: If return_norm is True, returns a tuple containing the normalized MPDO 
                            and the original norm.
        """
        norm = self.norm()
        if np.isclose(norm, 0):
            raise ValueError("Cannot normalize MPDO: computed norm is zero.")
        normalized_tensors = [tensor / norm**(1/self.num_sites) for tensor in self.tensors]
        if return_norm:
            return MPDO(normalized_tensors), norm
        return MPDO(normalized_tensors)
    
    def truncate(self, trunc_threshold: float, max_bond_dim: int, return_norm: bool = False) -> MPDO | tuple[MPDO, float]:
        """
        Perform SVD-based truncation on the MPDO to reduce bond dimensions while controlling truncation error.

        Args:
            trunc_threshold (float): Maximum allowed truncation error.
            max_bond_dim: Maximum number of singular values to retain.

        Returns:
            MPDO: The truncated MPDO.
            tuple[MPDO, float]: If return_norm is True, returns a tuple containing the normalized MPDO 
                            and the original norm.
        """
        self._fuse_phys_legs()
        result_tensors = [self.get_tensor(i) for i in range(self.num_sites)]
        self._split_phys_legs()
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
            if n == self.num_sites - 1:  
                norm = LA.norm(S)   
            S = np.diag(S[:chi_opt] / np.maximum(LA.norm(S[:chi_opt]), 1e-12))
            result_tensors[n] = V[:chi_opt,:].reshape(chi_opt, int(np.sqrt(d)), int(np.sqrt(d)), chi_r)
            result_tensors[n - 1] = ncon([result_tensors[n - 1], U[:,:chi_opt] @ S], [[-1, -2, 1], [1, -3]])
        result_tensors[0] = result_tensors[0].reshape(1, int(np.sqrt(d)), int(np.sqrt(d)), chi_opt) 
        if return_norm:
            return MPDO(result_tensors), norm 
        return MPDO(result_tensors)

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
    
    def _fuse_phys_legs(self):
        """
        Fuse the physical legs of the MPDO.
        """
        for i in range(self.num_sites):
            tensor = self.get_tensor(i)
            self.set_tensor(i, tensor.reshape(tensor.shape[0], self.phys_dim ** 2, tensor.shape[-1]))

    def _split_phys_legs(self):
        """
        Split the physical legs of the MPDO.
        """
        for i in range(self.num_sites):
            tensor = self.get_tensor(i)
            self.set_tensor(i, tensor.reshape(tensor.shape[0], self.phys_dim, self.phys_dim, tensor.shape[-1]))
    
    # ----------------------------- #
    # 4. SPECIAL METHODS
    # ----------------------------- #
    
    def __str__(self) -> str:
        """
        String representation of the MPDO.
        """
        return f"MPDO(num_sites={self.num_sites}, d={self.phys_dim}, bond_dims={self.bond_dims}, max_bond_dim={self.max_bond_dim})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the MPDO.
        """
        return self.__str__()