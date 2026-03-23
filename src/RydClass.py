import copy

from basic_operators import identity_matrix, pauli_x, proj_ket1
from src.MPOConstructors import MPOConstructors
from tensornetwork import ncon

class RydbergModel:
    """
    Class to model the evolution of a Rydberg Hamiltonian using MPOs.

    Attributes:
        num_sites (int): Number of sites in the system.
        omega (int): Rabi frequency, which controls the strength of the driving field.
        V (float): Interaction strength between Rydberg atoms.
        delta_t (float): Trotter time step.
        mpo_onsite (MPO): MPO for the on-site interaction.
        mpo_twobody (MPO): MPO for the two-body interaction.
        mpos (tuple[MPO]): MPOs involved in the evolution.
        complete_mpos (tuple[MPO]): Complete MPOs involved in the evolution.
    """

    # ----------------------------- #
    # 1. INITIALIZATION METHODS
    # ----------------------------- #

    def __init__(self, num_sites: int, omega: float, V: float, delta_t: float):
        """
        Initialize a RydbergModel instance using some parameters.
        """
        self.num_sites = num_sites
        self.omega = omega
        self.V = V
        self.delta_t = delta_t
        
        self.mpo_onsite = MPOConstructors(self.num_sites).on_site_observable(self.omega * self.delta_t, pauli_x, True)
        self.mpo_twobody = MPOConstructors(self.num_sites).two_body_correlation(self.V * self.delta_t, proj_ket1, True)        
        self.mpos = [self.mpo_onsite, self.mpo_twobody]        
        self.complete_mpos = copy.deepcopy(self.mpos)
        self._extend_mpos_with_identity()

    # ----------------------------- #
    # 2. PRIVATE METHODS
    # ----------------------------- #

    def _extend_mpos_with_identity(self):
        """
        Expands the MPOs with identity in the required additional spaces.
        """
        for i in range(2):
            mpo = self.complete_mpos[i]
            for n in range(self.num_sites):
                tensor = mpo.get_tensor(n)
                mpo.set_tensor(n, ncon([tensor, identity_matrix], [[-1,-2,-3,-5], [-4,-6]]).reshape(tensor.shape[0], tensor.shape[1], 4, 4))

    # ----------------------------- #
    # 3. SPECIAL METHODS
    # ----------------------------- #

    def __str__(self) -> str:
        """
        String representation of the RydbergModel.
        """
        return f"RydbergModel(num_sites={self.num_sites}, omega={self.omega}, V={self.V}, delta_t={self.delta_t})"
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the RydbergModel.
        """
        return self.__str__()