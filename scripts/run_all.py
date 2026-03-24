from scripts.plot_SCGF import main as plot_SCGF
from scripts.plot_activity import main as plot_activity
from scripts.plot_rate_function import main as plot_rate_function
from scripts.plot_critical_s import main as plot_critical_s
from scripts.plot_ensembles import main as plot_ensembles
from scripts.plot_phase_diagram import main as plot_phase_diagram
from scripts.plot_convergence import main as plot_convergence

def main():
    print("Running all plots...\n")

    plot_SCGF()
    plot_activity()
    plot_rate_function()
    plot_critical_s()
    plot_ensembles()
    plot_phase_diagram()
    plot_convergence()

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()