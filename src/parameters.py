from argparse import ArgumentParser

def parse_arguments(return_parser=False):
    parser = ArgumentParser(description="Run power method.")
    parser.add_argument('--num_sites', type=int, required=True, help="Number of sites in the system")
    parser.add_argument('--omega', type=float, default=1.0, help="Rabi frequency (default: 1.0)")
    parser.add_argument('--gamma', type=float, default=3.0, help="System-ancilla interaction strength (default: 3.0)")
    parser.add_argument('--V', type=float, required=True, help="Interaction strength")
    parser.add_argument('--Delta_t', type=float, default=1.25, help="Collision time step (default: 1.25)")
    parser.add_argument("--s", type=lambda x: round(float(x), 3), required=True, help="Bias parameter")
    parser.add_argument("--observable", type=str, default="00", help="Observable to measure (default: 00)")
    parser.add_argument('--num_trotter_steps', type=int, default=10, help="Trotter steps per iteration (default: 10)")
    parser.add_argument("--max_bond_dim", type=int, required=True, help="Maximum bond dimension")
    parser.add_argument('--trunc_threshold', type=float, default=1e-14, help="Threshold for truncation (default: 1e-14)")
    parser.add_argument('--max_iterations', type=int, required=True, help="Maximum number of iterations")
    parser.add_argument('--tolerance', type=float, default=1e-6, help="Convergence threshold (default: 1e-8)")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for storing results")
    parser.add_argument('--continue_simulation', type=bool, default=False, help="Continue previous simulation")

    if return_parser:
        return parser
    else:
        return parser.parse_args()