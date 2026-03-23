from src.JumpClass import Jump
from src.RydClass import RydbergModel

def initialize_simulation_parameters(params):
    delta_t = params.Delta_t / params.num_trotter_steps
    ryd = RydbergModel(num_sites=params.num_sites, omega=params.omega, V=params.V, delta_t=delta_t)
    jump = Jump(gamma=params.gamma, Delta_t=params.Delta_t, delta_t=delta_t)
    return ryd.complete_mpos, jump.interaction_tensor
