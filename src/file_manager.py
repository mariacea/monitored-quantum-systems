from datetime import datetime
from src.MPDOClass import MPDO
from src.utils import calculate_errors
from typing import Dict, List
import json, logging, os, pickle

def create_directory_structure(params):
    """
    Creates a directory structure for storing simulation results.

    Args:
        params (SimulationParams): The simulation parameters
    """
    max_bond_dim_dir = os.path.join(params.base_dir, f"num_sites_{params.num_sites}", f"V_{params.V:.3f}", f"s_{params.s:.3f}", f"max_bond_dim_{params.max_bond_dim}")
    os.makedirs(max_bond_dim_dir, exist_ok=True)

def save_results(params, eigenvalues: List[float], mpdo: MPDO, message: str = ""):
    """
    Saves simulation information.

    Args:
        params (SimulationParams): The simulation parameters.
        eigenvalues (List): List of eigenvalues from the iterations.
        mdpo (MPDO): Final MPDO. 
    """
    dir = os.path.join(params.base_dir, f"num_sites_{params.num_sites}", f"V_{params.V:.3f}", f"s_{params.s:.3f}", f"max_bond_dim_{params.max_bond_dim}")
    relative_errors = calculate_errors(eigenvalues)
    num_iterations = len(eigenvalues)
    timestamp = datetime.now().isoformat()

    # Load or initialize metadata
    metadata_path = os.path.join(dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "num_sites": params.num_sites,
            "V": params.V,
            "s": params.s,
            "max_bond_dim": params.max_bond_dim,
            "total_iterations": 0,
            "last_updated": None,
            "converged": False,
            "retake_count": 0,
            "execution_history": []
        }

    # Save iteration data
    iteration_data_path = os.path.join(dir, "iteration_data.json")
    if os.path.exists(iteration_data_path):
        with open(iteration_data_path, "r") as f:
            iteration_data = json.load(f)
    else:
        iteration_data = {"iterations": []}
    iteration_data["iterations"].extend([
        {
            "iteration_number": metadata["total_iterations"] + i + 1,
            "eigenvalue": eigenvalues[i],
            "relative_error": relative_errors[i],
        }
        for i in range(num_iterations)
    ])
    with open(iteration_data_path, "w") as f:
        json.dump(iteration_data, f, indent=4)

    # Update metadata
    metadata["total_iterations"] += num_iterations
    metadata["last_updated"] = timestamp
    metadata["converged"] = True if num_iterations < params.max_iterations else False
    if params.continue_simulation:
        metadata["retake_count"] += 1    
    metadata["execution_history"].append({
        "timestamp": timestamp,
        "continued": params.continue_simulation,
        "total_iterations_this_run": num_iterations
    })

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Save MPDO
    mpdo_path = os.path.join(dir, "steady_mpdo.pkl")
    with open(mpdo_path, "wb") as f:
        pickle.dump(mpdo, f)
    
    # Save log
    log_path = os.path.join(dir, "log.txt")
    with open(log_path, "a") as f:
        f.write(f"[INFO] Simulation for num_sites={params.num_sites}, V = {params.V:.3f}, s={params.s:.3f}, max_bond_dim={params.max_bond_dim}\n")
        f.write(f"[INFO] Timestamp: {timestamp}\n")
        if params.continue_simulation:
            f.write(f"[INFO] This calculation is a continuation of a previous run (Retake #{metadata['retake_count']}).\n")
        else:
            f.write(f"[INFO] This is a new calculation.\n")
        if message:
            f.write(f"[INFO] {message}\n")
        for i, (val, err) in enumerate(zip(eigenvalues, relative_errors)):
            error_str = f"{err:.14f}" if err is not None else "N/A"
            f.write(f"[INFO] Iteration {i+1}: eigenvalue={val}, relative_error={error_str}\n")
        f.write(f"[INFO] Total iterations this run: {num_iterations}\n")
        f.write(f"[INFO] ---------------------------------------------\n")

# def load_results(params, loading_mpdo=False) -> Dict | None:
def load_results(params, i, loading_mpdo=False) -> Dict | None:
    """
    Checks if the results for specific 'num_sites', 'V', 's' and 'max_bond_dim' already exist and loads them if available.

    Args:
        params (SimulationParams): The simulation parameters.

    Returns:
        Dict | None: A dictionary with the loaded data if it exists, or None if not found.
    """
    # dir = os.path.join(params.base_dir, f"num_sites_{params.num_sites}", f"V_{params.V:.3f}", f"s_{params.s:.3f}", f"max_bond_dim_{params.max_bond_dim}")
    dir = os.path.join(params.base_dir, f"num_sites_{params.num_sites}", f"V_{params.V:.3f}", f"s_{params.s:.3f}", f"T_{i}",f"max_bond_dim_{params.max_bond_dim}")
    metadata_path = os.path.join(dir, "metadata.json")
    # print(metadata_path)
    iteration_data_path = os.path.join(dir, "iteration_data.json")
    mpdo_path = os.path.join(dir, "steady_mpdo.pkl")
    if os.path.exists(metadata_path):
        logging.info(f"Results for num_sites={params.num_sites}, V={params.V:.3f}, s={params.s:.3f}, max_bond_dim={params.max_bond_dim} already exist. Loading...")
        result = {}

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                result["metadata"] = metadata
        except Exception as e:
            logging.error(f"Error loading metadata from {metadata_path}: {e}")

        logging.disable(logging.WARNING)

        try:
            with open(iteration_data_path, "r") as f:
                iteration_data = json.load(f)
                result["iteration_data"] = iteration_data
        except Exception:
            pass

        logging.disable(logging.NOTSET)

        if loading_mpdo == True:
            try:
                with open(mpdo_path, "rb") as f:
                    loaded_mpdo = pickle.load(f)
                    result["mpdo"] = loaded_mpdo
            except Exception as e:
                logging.warning(f"MPDO not found or failed to load from {mpdo_path}: {e}")

        return result

    return None