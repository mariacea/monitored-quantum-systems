from matplotlib.colors import to_rgba

gradient_blue = [
    "#E4F3F6","#B8E0E7","#82C9D6",
    "#4FB2C6","#2C9CB5","#1F7F95","#166370"
]

gradient_red = [
    "#FBE9E3","#F6CFC3","#F0B2A0",
    "#EA937D","#E37259","#C85D47","#9F4736"
]


color_map = {
    2.0: gradient_blue,
    5.875: gradient_red,
}

marker_map = {
    2.0: "d",
    5.875: "o",
}


def with_alpha(colors, alpha):
    return [to_rgba(c, alpha) for c in colors]

def get_colors(V, alpha=None):
    colors = color_map[V]
    if alpha is not None:
        return with_alpha(colors, alpha)
    return colors

def get_marker(V):
    return marker_map[V]