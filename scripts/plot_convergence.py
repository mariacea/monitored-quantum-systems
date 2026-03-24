import os
import json
import numpy as np
import matplotlib.pyplot as plt

from src.plot_style import set_plot_style
from plot_utils import create_figure
from plot_config import get_colors, get_marker
from matplotlib.ticker import ScalarFormatter


def plot_convergence(data, s, V, name):
    fig, ax = create_figure(6.0, 4.0, margin_cm=2.0)

    num_sites_range = np.array([4, 6, 8, 10, 20, 40, 60])
    max_bond_dim_range = np.array([8, 32, 64, 96])

    colors = get_colors(V)
    colors_soft = get_colors(V, alpha=0.3)
    marker = get_marker(V)

    for n, L in enumerate(num_sites_range):
        values = data[f"s_{s:.3f}"][f"V_{V:.3f}"][f"L_{L}"]["theta"]
        ax.plot(max_bond_dim_range, values, marker=marker, markersize=4.0, markeredgewidth=0.5, ls=':', lw=0.5, c=colors[n], markerfacecolor=colors_soft[n], markeredgecolor=colors[n])

    ax.set_xlabel(r"$D_\mathrm{max}$", fontsize=13)
    ax.set_ylabel(r"$\theta(s)$", fontsize=13)
    ax.set_xticks([8, 32, 64, 96])
    ax.tick_params(axis='both', labelsize=13)

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.offsetText.set_fontsize(13)

    plt.savefig(f'figures/{name}.pdf', bbox_inches='tight', dpi=300, transparent=True)

    plt.close(fig)

cases = [(-0.01, 5.875, "figs2_a"), ( 0.001, 5.875, "figs2_b"), ( 0.01, 5.875, "figs2_c"), (-0.01, 2.000, "figs2_d"), ( 0.001, 2.000, "figs2_e"), ( 0.01, 2.000, "figs2_f")]

def main():
    set_plot_style()

    os.makedirs("figures", exist_ok=True)

    with open('figures_data/convergence_data.json', 'r') as f:
        data = json.load(f)

    for s, V, name in cases:
        plot_convergence(data, s, V, name)