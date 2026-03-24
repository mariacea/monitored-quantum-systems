import matplotlib.pyplot as plt

def cm_to_inch(cm):
    return cm / 2.54

def create_figure(panel_width_cm, panel_height_cm, margin_cm=2.0):
    fig_width = cm_to_inch(panel_width_cm + 2 * margin_cm)
    fig_height = cm_to_inch(panel_height_cm + 2 * margin_cm)

    fig = plt.figure(figsize=(fig_width, fig_height))

    left = margin_cm / (panel_width_cm + 2 * margin_cm)
    bottom = margin_cm / (panel_height_cm + 2 * margin_cm)
    width = panel_width_cm / (panel_width_cm + 2 * margin_cm)
    height = panel_height_cm / (panel_height_cm + 2 * margin_cm)

    ax = fig.add_axes([left, bottom, width, height])

    return fig, ax