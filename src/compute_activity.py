from src.TensorNetworkSolver import TNsetup
from src.operators import initialize_simulation_parameters
from src.parameters import parse_arguments
from src.MPDOClass import MPDO
import numpy as np
import logging, json, os


args = parse_arguments()
system_evolution_op, ancilla_interaction_op = initialize_simulation_parameters(args)
setup = TNsetup(args.num_sites, args.observable, args.s, system_evolution_op, ancilla_interaction_op)

if np.abs(args.s) < 1e-10:
    piece = np.zeros([1, 2, 2, 1], dtype=complex)
    piece[0,:,:,0] = np.array([[0.5,0.0],[0.0,0.5]])
    steady_state = MPDO([piece for i in range(args.num_sites)])
else:
    logging.warning(f"The value of s is different from zero: activity cannot be computed in this way.")

mixed_state = setup.apply_collision_step(steady_state, args.num_trotter_steps, args.trunc_threshold, args.max_bond_dim)
mixed_state = setup._contract_ancilla_legs(mixed_state)
mixed_state, norm = mixed_state.truncate(args.trunc_threshold, args.max_bond_dim, return_norm=True)
lambda_value = (mixed_state.overlap(steady_state) * norm).real
theta_value = np.log(lambda_value) / args.num_sites
lambda_derivative_value = setup.find_eigenvalue_derivative(steady_state, args.num_trotter_steps, args.trunc_threshold, args.max_bond_dim)
activity = - lambda_derivative_value / (args.num_sites * lambda_value)
dir = os.path.join(args.base_dir, f"num_sites_{args.num_sites}", f"V_{args.V:.3f}", f"s_{args.s:.3f}", f"max_bond_dim_{args.max_bond_dim}")
metadata_path = os.path.join(dir, "metadata.json")
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading JSON at {metadata_path}. Creating new JSON.")
        metadata = {}

    metadata["activity"] = activity.real

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        f.write("\n")
else:
    print(f"The file {metadata_path} does not exist.")

