from src.TensorNetworkSolver import TNsetup
from src.operators import initialize_simulation_parameters
from src.parameters import parse_arguments
from src.MPDOClass import MPDO
from basic_operators import identity_matrix, ket0, ket1, flat, proj_ket0
from tensornetwork import ncon
import numpy as np
import json, os, time

# USEFUL!!!
# It computes the system trace
def compute_system_trace(rhom):
    L = rhom.num_sites
    trace = ncon([rhom.get_tensor(0)], [[-1, 1, 1, -2]])
    for i in range(1, L):
        trace = ncon([trace, rhom.get_tensor(i)], [[-1, 1], [1, 2, 2, -2]])
    return trace.item()

# USEFUL!!!
def compute_down_envs(left_env_m):
    L = left_env_m.num_sites
    down_envs = [ncon([left_env_m.get_tensor(L - 1), flat], [[-1, 1, 1, 2, -2], [2]])]
    for l in range(L - 2, 0, -1):
        down_envs.insert(0, ncon([down_envs[0], left_env_m.get_tensor(l), flat], [[3, -2], [-1, 1, 1, 2, 3], [2]]))
    return down_envs

# USEFUL!!!
def compute_n_t(rho_m):
    L = rho_m.num_sites
    down_envs = [ncon([rho_m.get_tensor(L - 1)], [[-1, 1, 1, -2]])]
    for l in range(L - 2, 0, -1):
        down_envs.insert(0, ncon([down_envs[0], rho_m.get_tensor(l)], [[2, -2], [-1, 1, 1, 2]]))
    up_envs = [ncon([rho_m.get_tensor(0)], [[-1, 1, 1, -2]])]
    for l in range(1, L - 1):
        up_envs.append(ncon([up_envs[-1], rho_m.get_tensor(l)], [[-1, 1], [1, 2, 2, -2]]))
    values = [ncon([rho_m.get_tensor(0), proj_ket0, down_envs[0]], [[-1, 1, 2, 3], [2, 1], [3, -2]]).item()]
    for l in range(1, L - 1):
        values.append(ncon([up_envs[l - 1], rho_m.get_tensor(l), proj_ket0, down_envs[l]], [[-1, 1], [1, 2, 3, 4], [3, 2], [4, -2]]).item())
    values.append(ncon([up_envs[-1], rho_m.get_tensor(L - 1), proj_ket0], [[-1, 1], [1, 2, 3, -2], [3, 2]]).item())   
    return values

# USEFUL!!!
def compute_trace_rho_m(rho_m):
    trace = ncon([rho_m.get_tensor(0)], [[-1, 1, 1, -2]])
    for l in range(1, rho_m.num_sites):
        trace = ncon([trace, rho_m.get_tensor(l)], [[-1, 1], [1, 2, 2, -2]])
    return trace.item()

# USEFUL!!!
def efficient_sample_2d(steady_state, T):
    total_norm = 1.0
    trajectory = []
    total_observables = []
    for t in range(1, T + 1):
        trajectory_t = []
        if t == 1:
            left_env = setup.apply_collision_step(steady_state, args.num_trotter_steps, args.trunc_threshold, args.max_bond_dim)
        else: 
            left_env = setup.apply_collision_step(left_env_m, args.num_trotter_steps, args.trunc_threshold, args.max_bond_dim)
        left_env_m = setup.compute_rhom(left_env)
        down_envs = compute_down_envs(left_env_m)

        # l = 0
        up_envs = []
        trace = ncon([left_env_m.get_tensor(0), down_envs[0], flat], [[-1, 1, 1, 2, 3], [3, -2], [2]]).item().real
        p = (ncon([left_env_m.get_tensor(0), down_envs[0], ket0.full().reshape(2)], [[-1, 1, 1, 2, 3], [3, -2], [2]]) / trace).real
        rn = np.random.rand()
        if rn < p:
            trajectory_t.append(0)
            op = ket0.full().reshape(2)
        else:
            trajectory_t.append(1)
            op = ket1.full().reshape(2)
        left_env_m.set_tensor(0, ncon([left_env_m.get_tensor(0), op], [[-1, -2, -3, 1, -4], [1]]))
        up_envs.append(ncon([left_env_m.get_tensor(0), identity_matrix], [[-1, 1, 2, -2], [2, 1]]))
        # l = 1, 2, ... args.num_sites - 2
        for l in range(1, args.num_sites - 1):
            trace = ncon([left_env_m.get_tensor(l), up_envs[l - 1], down_envs[l], flat], [[1, 2, 2, 3, 4], [-1, 1], [4, -2], [3]]).item().real
            p = (ncon([left_env_m.get_tensor(l), up_envs[l - 1], down_envs[l], ket0.full().reshape(2)], [[1, 2, 2, 3, 4], [-1, 1], [4, -2], [3]]) / trace).real
            rn = np.random.rand()
            if rn < p:
                trajectory_t.append(0)
                op = ket0.full().reshape(2)
            else:
                trajectory_t.append(1)
                op = ket1.full().reshape(2)
            left_env_m.set_tensor(l, ncon([left_env_m.get_tensor(l), op], [[-1, -2, -3, 1, -4], [1]]))
            up_envs.append(ncon([up_envs[-1], left_env_m.get_tensor(l), identity_matrix], [[-1, 1], [1, 2, 3, -2], [3, 2]]))
        # l = args.num_sites - 1

        trace = ncon([left_env_m.get_tensor(args.num_sites - 1), up_envs[args.num_sites - 2], flat], [[1, 2, 2, 3, -2], [-1, 1], [3]]).item().real
        p = (ncon([left_env_m.get_tensor(args.num_sites - 1), up_envs[args.num_sites - 2], ket0.full().reshape(2)], [[1, 2, 2, 3, -2], [-1, 1], [3]]) / trace).real
        rn = np.random.rand()
        if rn < p:
            trajectory_t.append(0)
            op = ket0.full().reshape(2)
        else:
            trajectory_t.append(1)
            op = ket1.full().reshape(2)
        left_env_m.set_tensor(args.num_sites - 1, ncon([left_env_m.get_tensor(args.num_sites - 1), op], [[-1, -2, -3, 1, -4], [1]]))
        up_envs.append(ncon([up_envs[-1], left_env_m.get_tensor(args.num_sites - 1), identity_matrix], [[-1, 1], [1, 2, 3, -2], [3, 2]]))
        trajectory.append(trajectory_t)
        left_env_m, norm = left_env_m.truncate(10e-14, 4 ** (args.num_sites // 2), return_norm=True)
        total_norm = total_norm * norm
        observables = compute_n_t(left_env_m)
        total_trace = compute_trace_rho_m(left_env_m).real
        normalized_observables = [observable.real / total_trace for observable in observables]
        total_observables.append(normalized_observables)
    trace = (compute_system_trace(left_env_m) * total_norm).real
    return trajectory, trace, total_observables

# ------------------------------------------------------------------------------------------------

parser = parse_arguments(return_parser=True)
parser.add_argument("--T", type=int, required=True, help="Parametro extra SOLO para este script")
parser.add_argument("--n_samples", type=int, required=True, help="Parametro extra SOLO para este script")
args = parser.parse_args()

system_evolution_op, ancilla_interaction_op = initialize_simulation_parameters(args)
setup = TNsetup(args.num_sites, args.observable, args.s, system_evolution_op, ancilla_interaction_op)

if np.abs(args.s) < 1e-10:
    piece = np.zeros([1, 2, 2, 1], dtype=complex)
    piece[0,:,:,0] = np.array([[0.5,0.0],[0.0,0.5]])
    steady_state = MPDO([piece for i in range(args.num_sites)])

T = args.T
histogram = dict()
for n in range(args.n_samples):
    print('Sample', n + 1)
    t0 = time.time()
    trajectory, trace, local_observables = efficient_sample_2d(steady_state, T)
    print('Local observables', local_observables)
    t1 = time.time()
    elapsed_minutes = (t1 - t0) / 60
    print(f"Tiempo en ejecutar efficient_sample_2d: {elapsed_minutes:.2f} minutos")
    key = str(trajectory)
    if key not in histogram:
        histogram[key] = {"count": 0,"observables": local_observables}
    histogram[key]["count"] += 1

max_bond_dim_dir = os.path.join(
    args.base_dir,
    f"num_sites_{args.num_sites}",
    f"V_{args.V:.3f}",
    f"s_{args.s:.3f}",
    f"T_{T}",
    f"max_bond_dim_{args.max_bond_dim}"
)
os.makedirs(max_bond_dim_dir, exist_ok=True)

metadata_path = os.path.join(max_bond_dim_dir, "metadata.json")

if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        try:
            existing_metadata = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Advertencia: metadata.json corrupto, se sobrescribirá desde cero.")
            existing_metadata = {}
else:
    existing_metadata = {}

existing_histogram = existing_metadata.get("histogram", {})

for key, value in histogram.items():
    if key in existing_histogram:
        existing_histogram[key]["count"] += value["count"]
    else:
        existing_histogram[key] = value

existing_metadata.update({
    "num_sites": args.num_sites,
    "V": args.V,
    "s": args.s,
    "max_bond_dim": args.max_bond_dim,
    "histogram": existing_histogram,
})

with open(metadata_path, "w") as f:
    json.dump(existing_metadata, f, indent=4)

print(f"✅ Metadata actualizada correctamente en {metadata_path}")
