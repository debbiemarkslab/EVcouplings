"""
Visualization of mutation effects

Authors:
  Thomas A. Hopf
  Anna G. Green (mutation_pymol_script generalization)
"""

from math import isnan
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from bokeh import plotting as bp
from bokeh.core.properties import value as bokeh_value
from bokeh.models import HoverTool

from evcouplings.couplings.model import CouplingsModel
from evcouplings.visualize.pairs import (
    secondary_structure_cartoon, find_secondary_structure_segments
)
from evcouplings.visualize.pymol import pymol_mapping
from evcouplings.mutate.calculations import split_mutants
from evcouplings.visualize.misc import rgb2hex, colormap
from evcouplings.utils.calculations import entropy_vector

AA_LIST_PROPERTY = "WFYPMILVAGCSTQNDEHRK"


def plot_mutation_matrix(source, mutant_column="mutant",
                         effect_column="prediction_epistatic",
                         conservation_column="column_conservation",
                         order=AA_LIST_PROPERTY,
                         min_value=None, max_value=None,
                         min_percentile=None, max_percentile=None,
                         show_conservation=False,
                         secondary_structure=None, engine="mpl",
                         **matrix_style):
    """
    Plot a single-substitution mutation matrix

    Parameters
    ----------
    source : evcouplings.couplings.CouplingsModel or pandas.DataFrame
        Plot single mutation matrix predicted using CouplingsModel,
        or effect data for single mutations DataFrame
    mutant_column : str, optional (default: "mutant")
        If using source dataframe, extract single mutations from this column.
        Mutations have to be in format A100V.
    effect_column : str, optional (default: "prediction_epistatic")
        If using source dataframe, extract mutation effect from this column.
        Effects must be numeric.
    conservation_column : str, optional (default: "column_conservation")
        If using source dataframe, extract column conservation information
        from this column. Conservation values must be between 0 and 1. To
        plot conservation, set show_conservation=True.
    order : str or list, optional (default: AA_LIST_PROPERTY)
        Reorder y-axis (substitutions) according to this parameter. If None,
        substitutions will be inferred from source, and sorted alphabetically
        if source is a DataFrame.
    min_value : float, optional (default: None)
        Threshold colormap at this minimum value. If None, defaults to
        minimum value in matrix; if max_value is also None, defaults to
        -max(abs(matrix))
    max_value : float, optional (default: None)
        Threshold colormap at this maximum value. If None, defaults to
        maximum value in matrix; if min_value is also None, defaults to
        max(abs(matrix))
    min_percentile : int or float, optional (default: None)
        Set min_value to this percentile of the effect distribution. Overrides
        min_value.
    max_percentile : int or float, optional (default: None)
        Set max_value to this percentile of the effect distribution. Overrides
        max_value.
    show_conservation : bool, optional (default: False)
        Plot positional conservation underneath matrix. Only possible for
        engine == "mpl".
    secondary_structure : dict or pd.DataFrame
        Secondary structure to plot above matrix.
        Can be a dictionary of position (int) to
        secondary structure character ("H", "E", "-"/"C"),
        or a DataFrame with columns "id" and "sec_struct_3state"
        (as returned by Chain.residues, and DistanceMap.residues_i
        and DistanceMap.residues_j). Only supported by engine == "mpl".
    engine : {"mpl", "bokeh"}
        Plot matrix using matplotlib (static, more visualization options)
        or with bokeh (interactive, less visualization options)
    **matrix_style : kwargs
        Will be passed on to matrix_base_mpl or matrix_base_bokeh as kwargs

    Returns
    -------
    matplotlib AxesSuplot or bokeh Figure
        Figure/Axes object. Display bokeh figure using show().
    """
    def _extract_secstruct(secondary_structure):
        """
        Extract secondary structure for plotting functions
        """
        # turn into dictionary representation if
        # passed as a DataFrame
        if isinstance(secondary_structure, pd.DataFrame):
            secondary_structure = dict(
                zip(
                    secondary_structure.id.astype(int),
                    secondary_structure.sec_struct_3state
                )
            )

        # make sure we only retain secondary structure
        # inside the range of the mutation matrix
        secondary_structure = {
            i: sstr for (i, sstr) in secondary_structure.items()
            if i in positions
        }

        secstruct_str = "".join(
            [secondary_structure.get(i, "-") for i in positions]
        )

        return secstruct_str

    conservation = None

    # test if we will extract information from CouplingsModel,
    # or from a dataframe with mutations
    if isinstance(source, CouplingsModel):
        matrix = source.smm()
        positions = source.index_list
        substitutions = source.alphabet
        wildtype_sequence = source.seq()

        if show_conservation:
            conservation = entropy_vector(source)
    else:
        # extract position, WT and subs for each mutant, and keep singles only
        source = split_mutants(
            source, mutant_column
        ).query("num_mutations == 1")

        # turn positions into numbers (may be strings)
        source.loc[:, "pos"] = pd.to_numeric(source.loc[:, "pos"]).astype(int)

        # same for effects, ensure they are numeric
        source.loc[:, effect_column] = pd.to_numeric(
            source.loc[:, effect_column], errors="coerce"
        )

        substitutions = sorted(source.subs.unique())

        # group dataframe to get positional information
        source_grp = source.groupby("pos").first().reset_index().sort_values(by="pos")
        positions = source_grp.pos.values
        wildtype_sequence = source_grp.wt.values

        if show_conservation:
            source_grp.loc[:, conservation_column] = pd.to_numeric(
                source_grp.loc[:, conservation_column], errors="coerce"
            )
            conservation = source_grp.loc[:, conservation_column].values

        # create mutation effect matrix
        matrix = np.full((len(positions), len(substitutions)), np.nan)

        # mapping from position/substitution into matrix
        pos_to_i = {p: i for i, p in enumerate(positions)}
        subs_to_j = {s: j for j, s in enumerate(substitutions)}

        # fill matrix with values
        for idx, r in source.iterrows():
            matrix[pos_to_i[r["pos"]], subs_to_j[r["subs"]]] = r[effect_column]

    # reorder substitutions
    if order is not None:
        matrix_final = np.full((len(positions), len(substitutions)), np.nan)
        substitutions_list = list(substitutions)

        # go through new order row by row and put in right place
        for i, subs in enumerate(order):
            if subs in substitutions:
                matrix_final[:, i] = matrix[:, substitutions_list.index(subs)]

        # set substitutions to new list
        substitutions = list(order)
    else:
        matrix_final = matrix

    # determine ranges for matrix colormaps
    # get effects without NaNs
    effects = matrix_final.ravel()
    effects = effects[np.isfinite(effects)]

    if min_percentile is not None:
        min_value = np.percentile(effects, min_percentile)

    if max_percentile is not None:
        max_value = np.percentile(effects, max_percentile)

    matrix_style["min_value"] = min_value
    matrix_style["max_value"] = max_value

    # extract secondary structure
    if secondary_structure is not None:
        secondary_structure_str = _extract_secstruct(secondary_structure)
    else:
        secondary_structure_str = None

    if engine == "mpl":
        return matrix_base_mpl(
            matrix_final, positions, substitutions,
            conservation=conservation,
            wildtype_sequence=wildtype_sequence,
            secondary_structure=secondary_structure_str,
            **matrix_style
        )
    elif engine == "bokeh":
        # cannot pass conservation for bokeh
        return matrix_base_bokeh(
            matrix_final, positions, substitutions,
            wildtype_sequence=wildtype_sequence,
            **matrix_style
        )
    else:
        raise ValueError(
            "Invalid plotting engine selected, valid options are: "
            "mpl, bokeh"
        )


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
        Threshold colormap at this minimum value. If None, defaults to
        minimum value in matrix; if max_value is also None, defaults to
        -max(abs(matrix))
    max_value : float, optional (default: None)
        Threshold colormap at this maximum value. If None, defaults to
        maximum value in matrix; if min_value is also None, defaults to
        max(abs(matrix))
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
    if max_value is None and min_value is None:
        max_value = np.nanmax(np.abs(matrix))
        min_value = -max_value
    elif min_value is None:
        min_value = np.nanmin(matrix)
    elif max_value is None:
        max_value = np.nanmax(matrix)

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
            if type(pos) is tuple:
                # label will be in format segment AA pos, eg B_1 A 151
                pos = "{} {} {}".format(pos[0], wt_symbol, pos[1])
            else:
                pos = "{} {}".format(wt_symbol, pos)
        else:
            wt_symbol = None
            if type(pos) is tuple:
                pos = " ".join(map(str, pos))
            else:
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

    TOOLS = "hover"
    height_factor = 12
    width_factor = 10

    # create lists of values for x- and y-axes, which will be
    # axis labels;
    # keep all of these as strings so we can have WT/substitution
    # symbol in the label
    if wildtype_sequence is None:
        if type(positions[0]) is tuple:
            positions = [" ".join(list(map(str, p))) for p in positions]
        else:
            positions = list(map(str, positions))
    else:
        if type(positions[0]) is tuple:
            positions = [
                "{} {} {}".format(p[0], wildtype_sequence[i], p[1])
                for i, p in enumerate(positions)
            ]
        else:
            positions = [
                "{} {}".format(wildtype_sequence[i], p)
                for i, p in enumerate(positions)
            ]

    substitutions = list(map(str, substitutions))

    p = bp.figure(
        title=title,
        x_range=positions, y_range=substitutions,
        x_axis_location="above", width=width_factor * len(positions),
        height=height_factor * len(substitutions),
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
    p.axis.major_label_text_font_size = "{}pt".format(label_size)
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
                    show_wt_char=True, label_filter=None, secondary_structure_style=None):
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
        Threshold colormap at this minimum value. If None, defaults to
        minimum value in matrix; if max_value is also None, defaults to
        -max(abs(matrix))
    max_value : float, optional (default: None)
        Threshold colormap at this maximum value. If None, defaults to
        maximum value in matrix; if min_value is also None, defaults to
        max(abs(matrix))
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
    show_wt_char : bool, optional (default: True)
        Display wild-type symbol in axis labels
    label_filter : function, optional (default: None)
        Function with one argument (integer) that determines if a certain position
        label will be printed (if label_filter(pos)==True) or not.
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
    if max_value is None and min_value is None:
        max_value = np.abs(matrix_masked).max()
        min_value = -max_value
    elif min_value is None:
        min_value = matrix_masked.min()
    elif max_value is None:
        max_value = matrix_masked.max()

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
    for i, pos in zip(x_range, positions):
        # filter labels, if selected
        if label_filter is not None and not label_filter(pos):
            continue

        # determine what position label should be
        if show_wt_char and wildtype_sequence is not None:
            wt_symbol = wildtype_sequence[i]
            if type(pos) is tuple and len(pos) == 2:
                # label will be in format segment AA pos, eg B_1 A 151
                label = "{} {} {}".format(pos[0], wt_symbol, pos[1])
            else:
                label = "{} {}".format(wt_symbol, pos)

        else:
            if type(pos) is tuple:
                label = " ".join(map(str, pos))
            else:
                label = str(pos)

        ax.text(
            i + LABEL_X_OFFSET, y_bottom_res, label,
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


def mutation_pymol_script(mutation_table, output_file,
                          effect_column="prediction_epistatic",
                          mutant_column="mutant", agg_func="mean",
                          cmap=plt.cm.RdBu_r, segment_to_chain_mapping=None):
    """
    Create a Pymol .pml script to visualize single mutation
    effects

    Parameters
    ----------
    mutation_table : pandas.DataFrame
        Table with mutation effects (will be filtered
        for single mutants)
    output_file : str
        File path where to store pml script
    effect_column : str, optional (default: "prediction_epistatic")
        Column in mutation_table that contains mutation effects
    mutant_column : str, optional (default: "mutant")
        Column in mutation_table that contains mutations
        (in format "A123G")
    agg_func : str, optional (default: "mean")
        Function used to aggregate single mutations into one
        aggregated effect per position (any pandas aggregation
        operation, including "mean", "min, "max")
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
            (default: plt.cm.RdBu_r)
        Colormap used to map mutation effects to colors
    segment_to_chain_mapping: str or dict(str -> str), optional (default: None)
        PDB chain(s) that should be targeted by line drawing

        * If None, residues will be selected
          py position alone, which may cause wrong assignments
          if multiple chains are present in the structure.

        * Different chains can be assigned for position
          if a dictionary that maps from segment (str) to PDB chain (str)
          is given.

    Raises
    ------
    ValueError
        If no single mutants contained in mutation_table
    ValueError
        If mutation_table contains a segment identifier not
        found in segment_to_chain_mapping
    """
    # split mutation strings
    t = split_mutants(mutation_table, mutant_column)

    # only pick single mutants
    t = t.query("num_mutations == 1")

    if len(t) == 0:
        raise ValueError(
            "mutation_table does not contain any single "
            "amino acid substitutions."
        )

    # add a segment column if missing
    if "segment" not in t.columns:
        t.loc[:, "segment"] = None

    with open(output_file, "w") as f:

        #handle each segment independently
        # have to fill NaNs with a string for groupby to work
        t = t.fillna("none")
        for segment_name, _t in t.groupby("segment"):

            if segment_to_chain_mapping is None:
                chain = None

            elif type(segment_to_chain_mapping) is str:
                chain = segment_to_chain_mapping

            elif segment_name not in segment_to_chain_mapping:
                raise ValueError(
                      "Segment name {} has no mapping to PyMOL "
                      "chain. Available mappings are: {}".format(
                          segment_name, segment_to_chain_mapping
                      )
                )
            else:
                chain = segment_to_chain_mapping[segment_name]

            # aggregate into positional information
            _t = _t.loc[:, ["pos", effect_column]].rename(
                columns={"pos": "i", effect_column: "effect"}
            )

            t_agg = _t.groupby("i").agg(agg_func).reset_index()
            t_agg.loc[:, "i"] = pd.to_numeric(t_agg.i).astype(int)

            # map aggregated effects to colors
            max_val = t_agg.effect.abs().max()
            mapper = colormap(-max_val, max_val, cmap)
            t_agg.loc[:, "color"] = t_agg.effect.map(mapper)
            t_agg.loc[:, "show"] = "spheres"

            if chain is not None:
                chain_sel = ", chain '{}'".format(chain)
            else:
                chain_sel = ""

            f.write("as cartoon{}\n".format(chain_sel))
            f.write("color grey80{}\n".format(chain_sel))

            pymol_mapping(t_agg, f, chain, atom="CA")
