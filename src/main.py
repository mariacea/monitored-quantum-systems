from src.parameters import parse_arguments
from src.simulation_runner import run_simulation

if __name__ == "__main__":
    args = parse_arguments()
    run_simulation(args)
