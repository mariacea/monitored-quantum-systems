import logging
import numpy as np
import os
import pickle
from src.MPDOClass import MPDO

def initialize_random_mpdo(params):
    message = f"Initializing a random MPDO"
    logging.info(message)
    tensors = []
    tensors.append(np.random.rand(1, 2, 2, 4))
    for i in range(1, params.num_sites - 1):
        dif = abs(params.num_sites - 1 - i)
        if dif > i:
            tensors.append(np.random.rand(min(4 ** i, params.max_bond_dim), 2, 2, min(4 ** (i+1), params.max_bond_dim)))
        elif dif < i:
            tensors.append(np.random.rand(min(4 ** (dif + 1), params.max_bond_dim), 2, 2, min(4 ** dif, params.max_bond_dim)))
        else:
            tensors.append(np.random.rand(min(4 ** i, params.max_bond_dim), 2, 2, min(4 ** i, params.max_bond_dim)))
    tensors.append(np.random.rand(4, 2, 2, 1))
    return MPDO(tensors).normalize(), message

def find_nearest_mpdo(params) -> tuple[MPDO, str] | None:
    """
    Search for the nearest MPDO corresponding to a given 'num_sites', 'V', 's', and 'max_bond_dim'.
    The function tries to find an MPDO with the closest **smaller** 'max_bond_dim'.
    If none is found, it searches for a nearby 's' value.
    If still none is found, it returns None.

    Args:
        params (SimulationParams): The simulation parameters.

    Returns:
        tuple[MPDO, str] | None: The nearest MPDO found and a log message, or None if not found.
    """
    num_sites_dir = os.path.join(params.base_dir, f"num_sites_{params.num_sites}")
    num_sites_V_dir = os.path.join(num_sites_dir, f"V_{params.V:.3f}")
    num_sites_V_s_dir = os.path.join(num_sites_V_dir, f"s_{params.s:.3f}")
    try:
        print('3')
        smaller_bond_dims = [
            int(folder.split("_")[3]) for folder in os.listdir(num_sites_V_s_dir)
            if folder.startswith("max_bond_dim_") and folder.split("_")[3].isdigit()
            and int(folder.split("_")[3]) < params.max_bond_dim
        ]
        print(smaller_bond_dims)
        if smaller_bond_dims:
            nearest_bond_dim = min(smaller_bond_dims, key=lambda x: abs(x - params.max_bond_dim))
            mpdo_path = os.path.join(num_sites_V_s_dir, f"max_bond_dim_{nearest_bond_dim}", "steady_mpdo.pkl")
            if os.path.exists(mpdo_path):
                message = f"Found MPDO for num_sites={params.num_sites}, V={params.V:.3f}, s={params.s:.3f}, max_bond_dim={nearest_bond_dim}, loading..."
                logging.info(message)
                try:
                    with open(mpdo_path, "rb") as f:
                        loaded_mpdo = pickle.load(f)
                    return loaded_mpdo, message
                except Exception as e:
                    logging.error(f"Error loading MPDO from {mpdo_path}: {e}")
    except Exception as e:
        logging.error(f"Error while searching for MPDO: {e}")
    try:
        s_values = [
            float(folder.split("_")[1]) for folder in os.listdir(num_sites_V_dir)
            if folder.startswith("s_")
        ]        
        if params.s > 0:
            s_values = [s for s in s_values if s > params.s]
        else:
            s_values = [s for s in s_values if s < params.s]
        closest_s = min(s_values, key=lambda x: abs(x - params.s)) if s_values else None
        if closest_s is not None:
            mpdo_path = os.path.join(num_sites_V_dir, f"s_{closest_s:.3f}", f"max_bond_dim_{params.max_bond_dim}", "steady_mpdo.pkl")
            print(mpdo_path)
            if os.path.exists(mpdo_path):
                message = f"Found MPDO for num_sites={params.num_sites}, V={params.V:.3f}, s={closest_s:.3f} and max_bond_dim={params.max_bond_dim}, loading..."
                logging.info(message)
                try:
                    with open(mpdo_path, "rb") as f:
                        loaded_mpdo = pickle.load(f)
                    return loaded_mpdo, message
                except Exception as e:
                    logging.error(f"Error loading MPDO from {mpdo_path}: {e}")
    except Exception as e:
        logging.error(f"Error while searching for alternative 's' values: {e}")
    logging.warning(f"No MPDO found as initial guess")
    return None