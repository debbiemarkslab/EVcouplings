"""
Visualization of mutation effects

Authors:
  Thomas A. Hopf
"""

from math import isnan
from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bokeh import plotting as bp
from bokeh.properties import value as bokeh_value
from bokeh.models import HoverTool

from evcouplings.visualize.pairs import (
    secondary_structure_cartoon, find_secondary_structure_segments
)
from evcouplings.visualize.misc import rgb2hex


def matrix_base_bokeh(matrix, positions, substitutions,
                      wildtype_sequence=None, label_size=8,
                      min_value=None, max_value=None,
                      colormap=plt.cm.RdBu_r, na_color="#bbbbbb",
                      title=None):
    """
    Bokeh-based interactive mutation matrix plotting. This is the base
    plotting function, see plot_mutation_matrix() for more convenient access.

    Parameters
    ----------
    matrix : np.array(float)
        2D numpy array with values for individual single mutations
        (first axis: position, second axis: substitution)
    positions : list(int) or list(str)
        List of positions along x-axis of matrix
        (length has to agree with first dimension of matrix)
    substitutions : list(str)
        List of substitutions along y-axis of matrix
        (length has to agree with second dimension of matrix)
    wildtype_sequence : str or list(str), optional (default: None)
        Sequence of wild-type symbols. If given, will indicate wild-type
        entries in matrix with a dot.
    label_size : int, optional (default: 8)
        Font size of x/y-axis labels.
    min_value : float, optional (default: None)
        Threshold colormap at this minimum value.
    max_value : float, optional (default: None)
        Threshold colormap at this maximum value.
    colormap : matplotlib colormap object, optional (default: plt.cm.RdBu_r)
        Maps mutation effects to colors of matrix cells.
    na_color : str, optional (default: "#bbbbbb")
        Color for missing values in matrix
    title : str, optional (default: None)
        If given, set title of plot to this value.

    Returns
    -------
    bokeh.plotting.figure.Figure
        Bokeh figure (for displaying or saving)
    """

    # figure out maximum and minimum values for color map
    if max_value is None or min_value is None:
        max_value = np.nanmax(np.abs(matrix))
        min_value = -max_value

    # use matplotlib colormaps to create color values,
    # set ranges based on given values
    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=colormap)

    # build list of values for plotting from source matrix
    pos_list = []
    subs_list = []
    color_list = []
    effect_list = []

    # go through values on x-axis (substitutions)
    for i, pos in enumerate(positions):
        if wildtype_sequence is not None:
            wt_symbol = wildtype_sequence[i]
            pos = "{} {}".format(wt_symbol, pos)
        else:
            wt_symbol = None
            pos = str(pos)

        # go through all values on y-axis (substitutions)
        for j, subs in enumerate(substitutions):
            pos_list.append(pos)
            subs_list.append(str(subs))

            cur_effect = matrix[i, j]
            if isnan(cur_effect):
                cur_effect_str = "n/a"
                color_list.append(na_color)
            else:
                cur_effect_str = "{:.2f}".format(cur_effect)
                color_list.append(
                    rgb2hex(*mapper.to_rgba(cur_effect))
                )

            # attach info if this is WT to WT self substitution
            if subs == wt_symbol:
                cur_effect_str += " (WT)"

            effect_list.append(cur_effect_str)

    source = bp.ColumnDataSource(
        data=dict(
            position=pos_list,
            substitution=subs_list,
            color=color_list,
            effect=effect_list,
        )
    )

    TOOLS = "resize,hover"
    height_factor = 12
    width_factor = 10

    # create lists of values for x- and y-axes, which will be
    # axis labels;
    # keep all of these as strings so we can have WT/substitution
    # symbol in the label
    if wildtype_sequence is None:
        positions = list(map(str, positions))
    else:
        positions = [
            "{} {}".format(wildtype_sequence[i], p)
            for i, p in enumerate(positions)
        ]

    substitutions = list(map(str, substitutions))

    p = bp.figure(
        title=title,
        x_range=positions, y_range=substitutions,
        x_axis_location="above", plot_width=width_factor * len(positions),
        plot_height=height_factor * len(substitutions),
        toolbar_location="left", tools=TOOLS
    )

    p.rect(
        "position", "substitution", 1, 1, source=source,
        color="color", line_color=None
    )

    # modify plot style
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = bokeh_value("{}pt".format(label_size))
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 2
    p.toolbar_location = None

    p.select_one(HoverTool).tooltips = [
        ('mutant', '@position @substitution'),
        ('effect', '@effect'),
    ]

    return p


def matrix_base_mpl(matrix, positions, substitutions, conservation=None,
                    secondary_structure=None, wildtype_sequence=None,
                    min_value=None, max_value=None,
                    ax=None, colormap=plt.cm.RdBu_r,
                    colormap_conservation=plt.cm.Oranges, na_color="#bbbbbb",
                    title=None, position_label_size=8, substitution_label_size=8,
                    show_colorbar=True, colorbar_indicate_bounds=False,
                    secondary_structure_style=None):
    """
    Matplotlib-based mutation matrix plotting. This is the base plotting function,
    see plot_mutation_matrix() for more convenient access.

    Parameters
    ----------
    matrix : np.array(float)
        2D numpy array with values for individual single mutations
        (first axis: position, second axis: substitution)
    positions : list(int) or list(str)
        List of positions along x-axis of matrix
        (length has to agree with first dimension of matrix)
    substitutions : list(str)
        List of substitutions along y-axis of matrix
        (length has to agree with second dimension of matrix)
    conservation : list(float) or np.array(float), optional (default: None)
        Positional conservation along sequence. Values must range
        between 0 (not conserved) and 1 (fully conserved). If given,
        will plot conservation along bottom of mutation matrix.
    secondary_structure : str or list(str), optional (default: None)
        Secondary structure for each position along sequence. If given,
        will draw secondary structure cartoon on top of matrix.
    wildtype_sequence : str or list(str), optional (default: None)
        Sequence of wild-type symbols. If given, will indicate wild-type
        entries in matrix with a dot.
    min_value : float, optional (default: None)
        Threshold colormap at this minimum value.
    max_value : float, optional (default: None)
        Threshold colormap at this maximum value.
    ax : Matplotlib axes object, optional (default: None)
        Draw mutation matrix on this axis. If None, new figure and axis
        will be created.
    colormap : matplotlib colormap object, optional (default: plt.cm.RdBu_r)
        Maps mutation effects to colors of matrix cells.
    colormap_conservation: matplotlib colormap object, optional (default: plt.cm.Oranges)
        Maps sequence conservation to colors of conservation vector plot.
    na_color : str, optional (default: "#bbbbbb")
        Color for missing values in matrix
    title : str, optional (default: None)
        If given, set title of plot to this value.
    position_label_size : int, optional (default: 8)
        Font size of x-axis labels.
    substitution_label_size : int, optional (default: 8)
        Font size of y-axis labels.
    show_colorbar : bool, optional (default: True)
        If True, show colorbar next to matrix.
    colorbar_indicate_bounds : bool, optional (default: False)
        If True, add greater-than/less-than signs to limits of colorbar
        to indicate that colors were thresholded at min_value/max_value
    secondary_structure_style : dict, optional (default: None)
        Pass on as **kwargs to evcouplings.visualize.pairs.secondary_structure_cartoon
        to determine appearance of secondary structure cartoon.

    Returns
    -------
    ax : Matplotlib axes object
        Axes on which mutation matrix was drawn
    """
    LINEWIDTH = 0.0
    LABEL_X_OFFSET = 0.55
    LABEL_Y_OFFSET = 0.45

    def _draw_rect(x_range, y_range, linewidth):
        r = plt.Rectangle(
            (min(x_range), min(y_range)),
            max(x_range) - min(x_range), max(y_range) - min(y_range),
            fc='None', linewidth=linewidth
        )
        ax.add_patch(r)

    matrix_width = matrix.shape[0]
    matrix_height = len(substitutions)

    # mask NaN entries in mutation matrix
    matrix_masked = np.ma.masked_where(np.isnan(matrix), matrix)

    # figure out maximum and minimum values for color map
    if max_value is None or min_value is None:
        max_value = np.abs(matrix_masked).max()
        min_value = -max_value

    # set NaN color value in colormaps
    colormap = deepcopy(colormap)
    colormap.set_bad(na_color)
    colormap_conservation = deepcopy(colormap_conservation)
    colormap_conservation.set_bad(na_color)

    # determine size of plot (depends on how much tracks
    # with information we will add)
    num_rows = (
        len(substitutions) +
        (conservation is not None) +
        (secondary_structure is not None)
    )

    ratio = matrix_width / float(num_rows)

    # create axis, if not given
    if ax is None:
        fig = plt.figure(figsize=(ratio * 5, 5))
        ax = fig.gca()

    # make square-shaped matrix cells
    ax.set_aspect("equal", "box")

    # define matrix coordinates
    # always add +1 because coordinates are used by
    # pcolor(mesh) as beginning and start of rectangles
    x_range = np.array(range(matrix_width + 1))
    y_range = np.array(range(matrix_height + 1))
    y_range_avg = range(-2, 0)
    x_range_avg = range(matrix_width + 1, matrix_width + 3)
    y_range_cons = np.array(y_range_avg) - 1.5

    # coordinates for text labels (fixed axis)
    x_left_subs = min(x_range) - 1
    x_right_subs = max(x_range_avg) + 1

    if conservation is None:
        y_bottom_res = min(y_range_avg) - 0.5
    else:
        y_bottom_res = min(y_range_cons) - 0.5

    # coordinates for additional annotation
    y_ss = max(y_range) + 2

    # 1) main mutation matrix
    X, Y = np.meshgrid(x_range, y_range)
    cm = ax.pcolormesh(
        X, Y, matrix_masked.T, cmap=colormap, vmax=max_value, vmin=min_value
    )
    _draw_rect(x_range, y_range, LINEWIDTH)

    # 2) mean column effect (bottom "subplot")
    mean_pos = np.mean(matrix_masked, axis=1)[:, np.newaxis]
    X_pos, Y_pos = np.meshgrid(x_range, y_range_avg)
    ax.pcolormesh(
        X_pos, Y_pos, mean_pos.T, cmap=colormap, vmax=max_value, vmin=min_value
    )
    _draw_rect(x_range, y_range_avg, LINEWIDTH)

    # 3) amino acid average (right "subplot")
    mean_aa = np.mean(matrix_masked, axis=0)[:, np.newaxis]
    X_aa, Y_aa = np.meshgrid(x_range_avg, y_range)
    ax.pcolormesh(X_aa, Y_aa, mean_aa, cmap=colormap, vmax=max_value, vmin=min_value)
    _draw_rect(x_range_avg, y_range, LINEWIDTH)

    # mark wildtype residues
    if wildtype_sequence is not None:
        subs_list = list(substitutions)

        for i, wt in enumerate(wildtype_sequence):
            # skip unspecified entries
            if wt is not None and wt != "":
                marker = plt.Circle(
                    (x_range[i] + 0.5, y_range[subs_list.index(wt)] + 0.5),
                    0.1, fc='k', axes=ax
                )
                ax.add_patch(marker)

    # put labels along both axes of matrix

    # x-axis (positions)
    for i, res in zip(x_range, positions):
        ax.text(
            i + LABEL_X_OFFSET, y_bottom_res, str(res),
            size=position_label_size,
            horizontalalignment='center',
            verticalalignment='top',
            rotation=90
        )

    # y-axis (substitutions)
    for j, subs in zip(y_range, substitutions):
        # put on lefthand side of matrix...
        ax.text(
            x_left_subs, j + LABEL_Y_OFFSET, subs,
            size=substitution_label_size,
            horizontalalignment='center',
            verticalalignment='center'
        )

        # ...and on right-hand side of matrix
        ax.text(
            x_right_subs, j + LABEL_Y_OFFSET, subs,
            size=substitution_label_size,
            horizontalalignment='center', verticalalignment='center'
        )

    # draw colorbar
    if show_colorbar:
        cb = plt.colorbar(
            cm, ticks=[min_value, max_value],
            shrink=0.3, pad=0.15 / ratio, aspect=8
        )

        if colorbar_indicate_bounds:
            symbol_min, symbol_max = u"\u2264", u"\u2265"
        else:
            symbol_min, symbol_max = "", ""

        cb.ax.set_yticklabels(
            [
                "{symbol} {value:>+{width}.1f}".format(
                    symbol=s, value=v, width=0
                ) for (v, s) in [(min_value, symbol_min), (max_value, symbol_max)]
            ]
        )
        cb.ax.xaxis.set_ticks_position("none")
        cb.ax.yaxis.set_ticks_position("none")
        cb.outline.set_linewidth(0)

    # plot secondary structure cartoon
    if secondary_structure is not None:
        # if no style given for secondary structure, set default
        if secondary_structure_style is None:
            secondary_structure_style = {
                "width": 0.8,
                "line_width": 2,
                "strand_width_factor": 0.5,
                "helix_turn_length": 2,
                "min_sse_length": 2,
            }

        start, end, sse = find_secondary_structure_segments(secondary_structure)
        secondary_structure_cartoon(
            sse, sequence_start=start, sequence_end=end, center=y_ss, ax=ax,
            **secondary_structure_style
        )

    # plot conservation
    if conservation is not None:
        conservation = np.array(conservation)[:, np.newaxis]
        cons_masked = np.ma.masked_where(np.isnan(conservation), conservation)
        X_cons, Y_cons = np.meshgrid(x_range, y_range_cons)
        ax.pcolormesh(
            X_cons, Y_cons, cons_masked.T, cmap=colormap_conservation, vmax=1, vmin=0
        )
        _draw_rect(x_range, y_range_cons, LINEWIDTH)

    # remove chart junk
    for line in ['top', 'bottom', 'right', 'left']:
        ax.spines[line].set_visible(False)

    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    if title is not None:
        ax.set_title(title)

    return ax
