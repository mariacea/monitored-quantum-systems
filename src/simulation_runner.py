from src.operators import initialize_simulation_parameters
from src.mpdo_utils import initialize_random_mpdo, find_nearest_mpdo
from src.file_manager import create_directory_structure, save_results, load_results
from src.logger import log_message
from src.TensorNetworkSolver import TNsetup
import numpy as np
from src.MPDOClass import MPDO
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_simulation(params):
    create_directory_structure(params)
    system_evolution_op, ancilla_interaction_op = initialize_simulation_parameters(params)
    setup = TNsetup(params.num_sites, params.observable, params.s, system_evolution_op, ancilla_interaction_op)
    existing_results = load_results(params, loading_mpdo=True)
    if existing_results is None:
        if np.abs(params.s) < 1e-10:
            piece = np.zeros([1, 2, 2, 1], dtype=complex)
            piece[0,:,:,0] = np.array([[0.5,0.0],[0.0,0.5]])
            steady_state = MPDO([piece for i in range(params.num_sites)])
            eigenvalues = [1.0]
        else:
            mixed_state, message = find_nearest_mpdo(params) or initialize_random_mpdo(params)
            eigenvalues, steady_state = setup.find_steady_state(mixed_state, params.max_iterations, params.tolerance, params.num_trotter_steps, params.trunc_threshold, params.max_bond_dim)
        save_results(params, eigenvalues, steady_state)
    else:
        if params.continue_simulation == True:
            log_message(f"Results exist for num_sites={params.num_sites}, s={params.s}. Continuing simulation...")
            mixed_state = existing_results['mpdo']
            eigenvalues, steady_state = setup.find_steady_state(mixed_state, params.max_iterations, params.tolerance, params.num_trotter_steps, params.trunc_threshold, params.max_bond_dim)
            save_results(params, eigenvalues, steady_state)
        else:

            log_message(f"Results exist for num_sites={params.num_sites}, s={params.s}. Loading...")
            steady_state = existing_results['mpdo']
