from plots.plot_SCGF import main as plot_SCGF
from plots.plot_activity import main as plot_activity
from plots.plot_rate_function import main as plot_rate_function
from plots.plot_critical_s import main as plot_critical_s
from plots.plot_ensembles import main as plot_ensembles
from plots.plot_phase_diagram import main as plot_phase_diagram
from plots.plot_convergence import main as plot_convergence

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