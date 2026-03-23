import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "legend.fontsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.color": "gray",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": True,
        "legend.loc": "best",
    })