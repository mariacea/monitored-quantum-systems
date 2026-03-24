import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from plot_utils import create_figure
from plot_config import get_colors, get_marker
from src.plot_style import set_plot_style


def main():
    set_plot_style()

    os.makedirs("figures", exist_ok=True)

    fig, ax = create_figure(5.0, 5.5, margin_cm=2.0)

    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.axvline(0, color='grey', linewidth=1, linestyle='--')

    with open('figures_data/SCGF_data.json', 'r') as f:
        data = json.load(f)

    s_range = np.linspace(-0.01, 0.01, 21)
    num_sites_range = np.array([4, 6, 8, 10, 20, 40, 60])
    V_values = np.array([5.875, 2.0])

    for n, L in enumerate(num_sites_range):
        for V in V_values:
            SCGF_large_values = data[f"SCGF_V_{V}_L_{L}_large"]
            SCGF_small_values = data[f"SCGF_V_{V}_L_{L}_small" ]
            colors = get_colors(V)
            colors_soft = get_colors(V, alpha=0.3)
            marker = get_marker(V)
            errors = [abs(lv - sv) for lv, sv in zip(SCGF_large_values, SCGF_small_values)]
            ax.plot(s_range, SCGF_large_values, marker=marker, markersize=4, markeredgewidth=0.5, ls=':', lw=0.8, c=colors[n], markerfacecolor=colors_soft[n], markeredgecolor=colors[n])
            ax.errorbar(s_range, SCGF_large_values, yerr=errors, linestyle='None', capsize=3, elinewidth=1, zorder=1, capthick=0.5, lw=0.5, c=colors[n])

    ax.set_xlabel(r"$s$", fontsize=17)
    ax.set_ylabel(r"$\theta(s)$", fontsize=17)
    ax.set_xticks([-0.01, 0.0, 0.01])
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.offsetText.set_fontsize(17)

    plt.savefig('figures/fig3_a.pdf', dpi=300, transparent=True)

    plt.close(fig)


if __name__ == "__main__":
    main()