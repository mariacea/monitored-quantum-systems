import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def create_custom_shading(lim_color, color, vmin, vmax):
    """
    lim_color = percent of interval with actual shading
    color = in RGB tuple
    vmin = usual vmin in pcolormesh
    vmax = usual vmax in pcolormesh
    """
    # create costume colormap
    cmap = colors.ListedColormap([(*color, 1 / (lim_color * 1000) * i) if i < lim_color * 1000 else (0, 0, 0, 1)
                                      for i in range(1000)])
    
    boundaries = np.arange(vmin, vmax, 1 / 1000)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    return cmap, norm

def create_strong_shading(color, vmin, vmax, power):
    
    n = 256
    alphas = np.linspace(0, 1, n)**power   # potencia <1 aumenta contraste en valores pequeños
    
    cmap = colors.ListedColormap([
        (*color, a) for a in alphas
    ])
    
    boundaries = np.linspace(vmin, vmax, n+1)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    
    return cmap, norm

def create_binary_shading(color, vmin, vmax):
    
    cmap = colors.ListedColormap([
        (1, 1, 1, 1),        # blanco para 0
        (*color, 1.0)        # color sólido para >0
    ])
    
    boundaries = [vmin, 1e-12, vmax]  # pequeño umbral >0
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    
    return cmap, norm
