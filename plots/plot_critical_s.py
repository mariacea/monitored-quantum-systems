import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from plot_utils import create_figure
from plot_config import with_alpha
from scipy.optimize import curve_fit
from src.plot_style import set_plot_style

def setup_axes(ax):
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')

def format_axes(ax):
    ax.set_xlabel(r"$s$", fontsize=17)
    ax.tick_params(axis='both', labelsize=17)

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0, 0))
    ax.xaxis.offsetText.set_fontsize(17)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.offsetText.set_fontsize(17)

def plot_scgf_cross(ax, data, V, num_sites_range, colors, colors_soft, marker_map, fit_indices, a0):

    setup_axes(ax)

    for n, L in enumerate(num_sites_range):
        marker = marker_map[L]

        s_range = data[f"s_range_zoom_V_{V}"]
        y_large = np.array(data[f"SCGF_positive_V_{V}_L_{L}_large"])
        y_small = np.array(data[f"SCGF_positive_V_{V}_L_{L}_small"])

        errors = np.abs(y_large - y_small)

        ax.plot(s_range, y_large, marker=marker, ms=4.7, mew=0.5, lw=0.0, c=colors[n], markerfacecolor=colors_soft[n], markeredgecolor=colors[n])
        ax.errorbar( s_range, y_large, yerr=errors, linestyle='None', capsize=3, elinewidth=1, capthick=0.5, c=colors[n])

        i0, i1 = fit_indices
        m, b = np.polyfit(s_range[i0:i1+1], y_large[i0:i1+1], 1)

        s_cross = -b / (m + a0)

        ax.scatter(s_cross, m*s_cross + b, marker='x', s=20, lw=0.5, c='black')

        x = np.linspace(s_cross, s_range[i1], 100)
        ax.plot(x, m*x + b, ls='--', lw=0.5, c='black')

    s_small = np.linspace(0, max(s_range), 50)
    ax.plot(s_small, -a0 * s_small, lw=0.5, c='grey')

    format_axes(ax)


def plot_inset(ax, x, y, sigma, num_sites_range, colors, colors_soft, marker_map):

    setup_axes(ax)

    def f(x, a, b):
        return a*x + b

    popt, pcov = curve_fit(f, x, y, sigma=sigma, absolute_sigma=True)
    a, b = popt
    sigma_b = np.sqrt(pcov[1, 1])

    for n, L in enumerate(num_sites_range):
        ax.plot(x[n], y[n], marker=marker_map[L], ms=4.7, mew=0.5, lw=0.0, c=colors[n], markerfacecolor=colors_soft[n], markeredgecolor=colors[n])
        ax.errorbar(x[n], y[n], yerr=sigma[n], linestyle='None', capsize=3, elinewidth=1, capthick=0.5, c=colors[n])

    x_fit = np.linspace(0, max(x), 100)
    ax.plot(x_fit, f(x_fit, a, b), '--', lw=0.5, c='black')

    ax.scatter(0, b, marker='x', s=10, lw=0.5, c='black')
    ax.errorbar(0, b, yerr=sigma_b, linestyle='None', capsize=2, elinewidth=0.5, c='black', lw=0.5)

    ax.set_xlabel(r'$1/L$', fontsize=17)
    ax.set_ylabel(r'$s^*$', fontsize=17)
    ax.set_xticks([1/60, 1/40, 1/20])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', labelsize=17)

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.offsetText.set_fontsize(17)

def main():

    set_plot_style()

    os.makedirs("figures", exist_ok=True)

    with open('figures/data/SCGF_positive_data.json', 'r') as f:
        data = json.load(f)

    with open('figures/data/activity_data.json', 'r') as f:
        activity_data = json.load(f)

    num_sites_range = np.array([20, 40, 60])
    colors = ["#6B8E23", "#7A3E9D", "#1F4E8C"]
    colors_soft = with_alpha(colors, 0.5)
    marker_map = {20: "o", 40: "D", 60: "s"}

    # =========================
    # FIGURE 3 
    # =========================

    # ---- fig3_d
    fig, ax = create_figure(5.0, 5.5, margin_cm=2.0)

    for V in [5.875, 2.0]:
        for n, L in enumerate(num_sites_range):

            s_range = data[f"s_range_V_{V}_L_{L}"]
            y_large = data[f"SCGF_positive_V_{V}_L_{L}_large"]
            y_small = data[f"SCGF_positive_V_{V}_L_{L}_small"]

            errors = np.abs(np.array(y_large) - np.array(y_small))

            ax.plot(s_range, y_large,
                    marker=marker_map[L], ms=4.7, lw=0.0,
                    c=colors[n],
                    markerfacecolor=colors_soft[n],
                    markeredgecolor=colors[n])

            ax.errorbar(s_range, y_large, yerr=errors,
                        linestyle='None', capsize=3,
                        elinewidth=1, capthick=0.5,
                        c=colors[n])

            if V == 2.0:
                ax.plot(data[f"s_range_V_{V}"],
                        data[f"bound_V_{V}"],
                        c='grey', lw=0.5)

    format_axes(ax)

    plt.savefig('figures/fig3_d.pdf', dpi=300, transparent=True)
    plt.close(fig)

    # ---- fig3_e
    fig, ax = create_figure(5.0, 5.5, margin_cm=2.0)

    a0_activity = activity_data[f"activity_V_{5.875}_L_60_large"][10]

    plot_scgf_cross(ax, data, 5.875, num_sites_range, colors, colors_soft, marker_map, fit_indices=(0, 1), a0=a0_activity)

    plt.savefig('figures/fig3_e.pdf', dpi=300, transparent=True)
    plt.close(fig)

    # ---- fig3_e_inset
    fig, ax = create_figure(5.5/3, 4.5/4, margin_cm=2.0/5)

    x = np.array([1/20, 1/40, 1/60])
    y = np.array(data["sc_values_V_5.875"])
    sigma = np.array(data["error_values_V_5.875"])

    plot_inset(ax, x, y, sigma, num_sites_range, colors, colors_soft, marker_map)

    plt.savefig('figures/fig3_e_inset.pdf', dpi=300, transparent=True)
    plt.close(fig)

    # =========================
    # SUPPLEMENTAL FIGURES
    # =========================

    configs = [{"V": 3.8, "fit": (5, 6), "name": "figs3_a"}, {"V": 4.3, "fit": (1, 2), "name": "figs3_b"}, {"V": 5.0, "fit": (2, 3), "name": "figs3_c"}, {"V": 6.8, "fit": (1, 2), "name": "figs3_d"},]

    for cfg in configs:

        V = cfg["V"]
        fit = cfg["fit"]
        name = cfg["name"]

        # ---- main plot
        fig, ax = create_figure(6.5, 4.5, margin_cm=2.0)

        a0 = - data[f"bound_V_{V}"][-1] / data[f"s_range_V_{V}"][-1]

        plot_scgf_cross(ax, data, V, num_sites_range, colors, colors_soft, marker_map, fit_indices=fit, a0=a0)

        plt.savefig(f'figures/{name}.pdf', dpi=300, transparent=True)
        plt.close(fig)

        # ---- inset
        fig, ax = create_figure(5.5/3, 4.5/4, margin_cm=2.0/5)

        x = np.array(data[f"1/L_values_V_{V}"])
        y = np.array(data[f"sc_values_V_{V}"])
        sigma = np.array(data[f"error_values_V_{V}"])

        plot_inset(ax, x, y, sigma, num_sites_range, colors, colors_soft, marker_map)

        plt.savefig(f'figures/{name}_inset.pdf', dpi=300, transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    main()