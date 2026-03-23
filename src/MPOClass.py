from __future__ import annotations

from typing import List
import numpy as np
from tensornetwork import ncon

class MPO:
    """
    Class for Matrix Product Operators (MPO), a representation of quantum many-body operators.

    Attributes:
        tensors (List[np.ndarray]): Tensors defining the MPS.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, tensors: List[np.ndarray]):
        """
        Initialize an MPO object using a list of tensors.

        Args:
            tensors (List): List of tensors defining the MPO, each of shape (bond_dim_left, bond_dim_right, phys_dim, phys_dim).
            num_sites (int): Number of sites in the MPO.
            phys_dim (int): Physical dimension of the MPO.
            bond_dims (List[int]): List of the bond dimensions in the MPO.
            max_bond_dim (int): Maximum bond dimension of the MPO.
        """
        self.tensors = tensors

        self.num_sites = len(tensors)
        self.phys_dim = tensors[0].shape[2] 
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
            bond_dims.append(self.tensors[i].shape[1])
        max_bond_dim = max(bond_dims)        
        return bond_dims, max_bond_dim
    
    def _check_bond_dimension_consistency(self):
        """
        Check that the bond dimensions between adjacent tensors are consistent.
        """
        for i in range(self.num_sites - 1):
            bond_dim_right = self.tensors[i].shape[1]
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

    def apply_mpo(self, mpo: MPO) -> MPO:
        """
        Apply an MPO operator to the MPO and return the resulting MPO.

        Args:
            mpo (MPO): The MPO to be applied.

        Returns:
            MPO: The new MPO after applying the MPO.
        """
        if self.num_sites != mpo.num_sites:
            raise ValueError("MPOs must have the same number of sites")
        result_tensors = []
        for i in range(self.num_sites):
            tensor1 = self.get_tensor(i); shape1 = tensor1.shape
            tensor2 = mpo.get_tensor(i); shape2 = tensor2.shape
            result_tensor = ncon([tensor1,tensor2], [[-1, -3, 1, -6], [-2, -4, -5, 1]])
            result_tensors.append(result_tensor.reshape(*[shape1[j]*shape2[j] for j in range(2)],shape1[2],shape1[3]))
        return MPO(result_tensors)
    
    # ----------------------------- #
    # 4. SPECIAL METHODS
    # ----------------------------- #
    
    def __str__(self) -> str:
        """
        String representation of the MPO.
        """
        return f"MPO(num_sites={self.num_sites}, d={self.phys_dim}, bond_dims={self.bond_dims}, max_bond_dim={self.max_bond_dim})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the MPO.
        """
        return self.__str__()