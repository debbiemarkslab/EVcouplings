"""
Visualization of evolutionary couplings (contact maps etc.)

Authors:
  Thomas A. Hopf
"""

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evcouplings.visualize.pymol import (
    pymol_pair_lines, pymol_mapping
)

# default plotting styles
STYLE_EC = {
    "edgecolor": "none",
    "c": "black",
    "s": 80,
}

STYLE_CONTACT = {
    "edgecolor": "none",
    "c": "#b6d4e9",
    "s": 100,
}

STYLE_CONTACT_BRIGHT = {
    "edgecolor": "none",
    "c": "#d9e7f5",
    "s": 100,
}

STYLE_CONTACT_MULTIMER = {
    "edgecolor": "none",
    "c": "#fc8c3b",
    "alpha": 0.3,
    "s": 100,
}

STYLE_SECSTRUCT = {
    "helix_turn_length": 2,
    "strand_width_factor": 0.5,
    "min_sse_length": 2,
    "width": 1,
    "line_width": 2,
    "helix_color": "k",
    "strand_color": "k",
    "coil_color": "k",
}


def find_boundaries(boundaries, ecs, monomer, multimer, symmetric):
    """
    Identify axis boundaries for contact map plots

    Parameters
    ----------
    boundaries : {"union", "intersection", "ecs", "structure"} or tuple
             or list(tuple, tuple)
        Set axis range (min/max) of contact map as follows:

        * "union": Positions either in ECs or 3D structure
        * "intersection": Positions both in ECs and 3D structure
        * "ecs": Positions in ECs
        * "structure": Positions in 3D structure
        * tuple(float, float): Specify upper/lower bound manually
        * [(float, float), (float, float)]: Specify upper/lower bounds
          for both x-axis (first tuple) and y-axis (second tuple)

    ecs : pandas.DataFrame
        Table of evolutionary couplings to plot (using columns
        "i" and "j")
    monomer : evcouplings.compare.distances.DistanceMap
        Monomer distance map (intra-chain distances)
    multimer : evcouplings.compare.distances.DistanceMap
        Multimer distance map (multimeric inter-chain distances)
    symmetric : bool
        Sets if distance maps and ECs are symmetric (intra-chain or homomultimer),
        or not (inter-chain).

    Returns
    -------
    (min_x, max_x) : (float, float)
        First and last position on x-axis
    (min_y, max_y) : (float, float)
        First and last position on y-axis
    """
    def _find_pos(axis):
        """
        Find first and last index along a single contact map axis
        """
        # determine what sets of positions are for ECs/contact maps
        ec_pos = set()
        monomer_pos = set()
        multimer_pos = set()

        # need to merge i and j here if symmetric
        if ecs is not None:
            if symmetric:
                ec_pos = set(ecs.i.astype(int)).union(ecs.j.astype(int))
            else:
                ec_pos = set(getattr(ecs, axis).astype(int))

        if monomer is not None:
            monomer_pos = set(
                getattr(monomer, "residues_" + axis).id.astype(int)
            )

        if multimer is not None:
            multimer_pos = set(
                getattr(multimer, "residues_" + axis).id.astype(int)
            )

        structure_pos = monomer_pos.union(multimer_pos)

        # maximum ranges spanned by structure or ECs
        # if any of the sets is not given, revert to
        # the other set of positions in else case
        # (in these cases, union and intersection will
        # be trivially the one set that is actually defined)
        if len(ec_pos) > 0:
            min_ec, max_ec = min(ec_pos), max(ec_pos)
        else:
            min_ec, max_ec = min(structure_pos), max(structure_pos)

        if len(structure_pos) > 0:
            min_struct, max_struct = min(structure_pos), max(structure_pos)
        else:
            min_struct, max_struct = min(ec_pos), max(ec_pos)

        # determine and set plot boundaries
        if boundaries == "union":
            min_val = min(min_ec, min_struct)
            max_val = max(max_ec, max_struct)
        elif boundaries == "intersection":
            min_val = max(min_ec, min_struct)
            max_val = min(max_ec, max_struct)
        elif boundaries == "ecs":
            min_val = min_ec
            max_val = max_ec
        elif boundaries == "structure":
            min_val = min_struct
            max_val = max_struct
        else:
            raise ValueError(
                "Not a valid value for boundaries: {}".format(
                    boundaries
                )
            )

        return min_val, max_val

    # check first if range is specified manually
    if isinstance(boundaries, tuple):
        if len(boundaries) != 2:
            raise ValueError(
                "boundaries must be a tuple with 2 elements (min, max)."
            )
        min_x, max_x = boundaries
        min_y, max_y = boundaries
    elif isinstance(boundaries, list):
        if len(boundaries) != 2 or len(boundaries[0]) != 2 or len(boundaries[1]) != 2:
            raise ValueError(
                "boundaries must be a list of 2 tuples with 2 elements "
                "[(min_x, max_x), (min_y, max_y)]."
            )
        min_x, max_x = boundaries[0]
        min_y, max_y = boundaries[1]
    else:
        min_x, max_x = _find_pos("i")
        min_y, max_y = _find_pos("j")

    return (min_x, max_x), (min_y, max_y)


def plot_contact_map(ecs=None, monomer=None, multimer=None,
                     distance_cutoff=5, secondary_structure=None,
                     show_secstruct=True, scale_sizes=True,
                     ec_style=STYLE_EC, monomer_style=STYLE_CONTACT,
                     multimer_style=STYLE_CONTACT_MULTIMER,
                     secstruct_style=STYLE_SECSTRUCT,
                     margin=5, invert_y=True, boundaries="union",
                     symmetric=True, ax=None):
    """
    Wrapper for simple contact map plots with optional
    multimer contacts (can also handle non-symmetric inter-chain
    contacts). For full flexibility, compose your own
    contact map plot using the functions used below.

    Parameters
    ----------
    ecs : pandas.DataFrame
        Table of evolutionary couplings to plot (using columns
        "i" and "j"). Can contain additional columns "color"
        and "size" to assign these individual properties in the
        plot for each plotted pair (i, j). If all values of
        "size" are <= 1, they will be treated as a fraction
        of the point size defined in ec_style, and rescaled
        if scale_sizes is True.
    monomer : evcouplings.compare.distances.DistanceMap
        Monomer distance map (intra-chain distances)
    multimer : evcouplings.compare.distances.DistanceMap
        Multimer distance map (multimeric inter-chain distances)
    distance_cutoff : float, optional (default: 5)
        Pairs with distance <= this cutoff are considered
        residue pair contacts
    secondary_structure : dict or pandas.DataFrame
        Secondary structure to be displayed on both axis
        (if not given, will try to extract from monomer
        distance map). For format of dict or DataFrame,
        see documentation of plot_secondary_structure().
        If symmetric is False, this has to be a two-element
        tuple containing the secondary structures for the
        x-axis and the y-axis, respectively.
    show_secstruct : bool, optional (default: True)
        Draw secondary structure on both axes (either
        passed in explicitly using secondary_structure,
        or extracted from monomer distancemap)
    scale_sizes : bool, optional (default: True)
        Rescale sizes of points on scatter plots as well as
        secondery structure plotting width based on
        overall size of protein
    ec_style : dict, optional (default: STYLE_EC)
        Style for EC plotting (kwargs to plt.scatter)
    monomer_style : dict, optional (default: STYLE_CONTACT)
        Style for monomer contact plotting (kwargs to plt.scatter)
    multimer_style : dict, optional (default: STYLE_CONTACT_MULTIMER)
        Style for multimer contact plotting (kwargs to plt.scatter)
    secstruct_style : dict, optional (default: STYLE_SECSTRUCT)
        Style for secondary structure plotting
        (kwargs to secondary_structure_cartoon())
    margin : int, optional (default: 5)
        Space to add around contact map
    invert_y : bool, optional (default: True)
        Invert the y axis of the contact map so both sequences
        run from N -to C- terminus starting from top left corner
    boundaries : {"union", "intersection", "ecs", "structure"} or tuple or list(tuple, tuple), optional (default: "union")
        Set axis range (min/max) of contact map as follows:

        * "union": Positions either in ECs or 3D structure
        * "intersection": Positions both in ECs and 3D structure
        * "ecs": Positions in ECs
        * "structure": Positions in 3D structure
        * tuple(float, float): Specify upper/lower bound manually
        * [(float, float), (float, float)]: Specify upper/lower bounds
          for both x-axis (first tuple) and y-axis (second tuple)

    symmetric : bool, optional (default: True)
        Sets if distance maps and ECs are symmetric (intra-chain or homomultimer),
        or not (inter-chain).
    ax : Matplotlib Axes object, optional (default: None)
        Axes the plot will be drawn on
    """
    if ecs is None and monomer is None and multimer is None:
        raise ValueError(
            "Need to specify at least one of ecs, monomer or multimer"
        )

    if ax is None:
        ax = plt.gca()

    # figure out how to set contact map boundaries
    (min_x, max_x), (min_y, max_y) = find_boundaries(
        boundaries, ecs, monomer, multimer, symmetric
    )

    set_range(
        x=(min_x, max_x), y=(min_y, max_y),
        ax=ax, margin=margin, invert_y=invert_y
    )

    # enable rescaling of points and secondary structure if necessary
    if scale_sizes:
        scale_func = lambda x: scale(x, ax=ax)
    else:        
        scale_func = lambda x: x

    # plot monomer contacts
    # (distance maps will automatically be symmetric for
    # intra/homomultimer, so do not request mirroring)
    if monomer is not None:
        plot_pairs(
            monomer.contacts(distance_cutoff),
            symmetric=False, style=scale_func(monomer_style),
            ax=ax
        )

    # plot multimer contacts
    # (distance maps will automatically be symmetric for
    # intra/homomultimer, so again do not request mirroring)
    if multimer is not None:
        plot_pairs(
            multimer.contacts(distance_cutoff),
            symmetric=False, style=scale_func(multimer_style),
            ax=ax
        )

    # plot ECs
    # (may be symmetric or not, depending on use case)
    if ecs is not None:
        plot_pairs(
            ecs, symmetric=symmetric, style=scale_func(ec_style), ax=ax
        )

    # plot secondary structure
    if show_secstruct:
        # if secondary structure given explicitly, use it
        if secondary_structure is not None:
            if symmetric:
                secstruct_i = secondary_structure
                secstruct_j = secondary_structure
            else:
                if (not isinstance(secondary_structure, tuple) or
                        len(secondary_structure) != 2):
                    raise ValueError(
                        "When symmetric is False, secondary structure must "
                        "be a tuple (secstruct_i, secstruct_j)."
                    )
                secstruct_i, secstruct_j = secondary_structure

            plot_secondary_structure(
                secstruct_i, secstruct_j,
                style=scale_func(secstruct_style), ax=ax
            )
        else:
            # otherwise, see if we can extract it from monomer
            # distance map
            try:
                plot_secondary_structure(
                    monomer.residues_i, monomer.residues_j,
                    style=scale_func(secstruct_style), ax=ax
                )
            except AttributeError:
                # DataFrame has no secondary structure, cannot do anything here
                # (this happens when merging across multiple distance maps)
                pass


def plot_pairs(pairs, symmetric=False, ax=None, style=None):
    """
    Plot list of pairs (ECs/contacts)

    Parameters
    ----------
    pairs : pandas.DataFrame
        DataFrame with coordinates to plot
        (taken from columns i and j). If there
        are columns "color" and "size", these
        will be used to assign individual colors
        and sizes to the dots in the scatter plot.
        If sizes are all <= 1 and "s" is present as a
        key in style, values will be treated as fraction
        of "s".
    symmetric : bool, optional (default: False)
        If true, for each pair (i, j) also plot
        pair (j, i). This is for cases where
        the input list pairs does not contain
        both pairs.
    ax : matplotlib Axes object
        Axes to plot on
    style : dict
        Parameters to style pair scatter plot
        (passed as **kwargs to matplotlib plt.scatter())

    Returns
    -------
    paths : list of PathCollection
        Scatter plot paths drawn by this function
    """
    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")

    if style is None:
        style = {}

    if "color" in pairs.columns:
        style["c"] = pairs.loc[:, "color"].values

    if "size" in pairs.columns:
        # if all sizes <= 1, treat as fraction
        if len(pairs.query("size > 1")) == 0 and "s" in style:
            style["s"] *= pairs.loc[:, "size"].values
        # otherwise take as actual value
        else:
            style["s"] = pairs.loc[:, "size"].values

    path1 = ax.scatter(
        pairs.i.astype(int),
        pairs.j.astype(int),
        **style
    )

    paths = [path1]

    if symmetric:
        path2 = ax.scatter(
            pairs.j.astype(int),
            pairs.i.astype(int),
            **style
        )
        paths.append(path2)

    return paths


def set_range(pairs=None, symmetric=True, x=None, y=None,
              ax=None, margin=0, invert_y=True):
    """
    Set axes ranges for contact map based
    on minimal/maximal values in pair list

    Parameters
    ----------
    pairs : pandas.DataFrame, optional (default: None)
        DataFrame with pairs (will be extracted
        from columns named "i" and "j" and
        converted to integer values and used
        to define x- and y-axes, respectively).
        If None, if x and y have to be specified.
    symmetric : bool, optional (default:True)
        If true, will define range on joint set
        of values in columns i and j, resulting
        in a square contact map
    x : tuple(float, float), optional (default: None)
        Set x-axis range with this range (min, max).
        Will be extended by margin, and overrides any
        value for x-axis derived using pairs.
    y : tuple(float, float), optional (default: None)
        Set y-axis range with this range (min, max)
        Will be extended by margin, and overrides any
        value for y-axis derived using pairs.
    ax : matplotlib Axes object
        Axes for which plot range will be changed
    margin : int, optional (default: 0)
        Add margin of this size around the actual
        range defined by the data
    invert_y : bool, optional (default: True)
        Invert y-axis of contact map

    Returns
    -------
    x_range : tuple(int, int)
        Set range for x-axis
    y_range : tuple(int, int)
        Set range for y-axis

    Raises
    ------
    ValueError
        If any axis range remains unspecified
    """
    if ax is None:
        ax = plt.gca()

    x_range, y_range = None, None

    # infer plot range from data
    if pairs is not None:
        i = pairs.i.astype(int)
        j = pairs.j.astype(int)

        if symmetric:
            x_range = (
                min(i.min(), j.min()) - margin,
                max(i.max(), j.max()) + margin
            )
            y_range = x_range
        else:
            x_range = (i.min() - margin, i.max() + margin)
            y_range = (j.min() - margin, j.max() + margin)

    # Override with user-specified values
    if x is not None:
        x_range = (x[0] - margin, x[1] + margin)

    if y is not None:
        y_range = (y[0] - margin, y[1] + margin)

    if x_range is None or y_range is None:
        raise ValueError(
            "Axis remained unspecified (make sure to either "
            "set pairs pr x_range/y_range) :"
            " x: {} y: {}".format(
                x_range, y_range
            )
        )

    # maintain axis inversion
    # (which gets undone by setting x/ylim)
    inverted_x = ax.xaxis_inverted()
    inverted_y = ax.yaxis_inverted()

    # set new range
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # recover axis orientation
    if inverted_x:
        ax.invert_xaxis()

    if inverted_y or (invert_y and not inverted_y):
        ax.invert_yaxis()

    # make sure axis ticks are in right spot
    ax.yaxis.set_ticks_position("left")
    if ax.yaxis_inverted():
        ax.xaxis.set_ticks_position("top")
    else:
        ax.xaxis.set_ticks_position("bottom")

    return x_range, y_range


def scale(style, ax=None):
    """
    Scale size of drawn elements based on size
    of contact map plot
    """
    if ax is None:
        ax = plt.gca()

    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    L = max(
        abs(x_range[1] - x_range[0]),
        abs(y_range[1] - y_range[0])
    )
    style = deepcopy(style)

    # dot size
    if "s" in style.keys():
        style["s"] = style["s"]**2 / L

    # secondary structure width
    if "width" in style.keys():
        style["width"] = style["width"] * L / 100

    return style


def plot_secondary_structure(secstruct_i, secstruct_j=None, ax=None, style=None, margin=None):
    """
    Plot secondary structure along contact map.

    Note: this function should only be used *after* the
    orientation of the axes of the plot has been set.

    Parameters
    ----------
    secstruct_i : dict or pd.DataFrame
        Secondary structure for x-axis of plot.
        Can be a dictionary of position (int) to
        secondary structure character ("H", "E", "C", "-"),
        or a DataFrame with columns "id" and "sec_struct_3state"
        (as returned by Chain.residues, and DistanceMap.residues_i
        and DistanceMap.residues_j).
    secstruct_j : dict or pd.DataFrame, optional (default: None)
        Secondary structure for y-axis of plot.
        See secstruct_i for possible values.
        If None, will use secstruct_i for y-axis too
        (assuming symmetric contact map).
    ax : matplotlib Axes object
        Axis to draw secondary structure on
    style : dict, optional (default: None)
        Parameters that will be passed on as kwargs
        to secondary structure drawing function
    margin : int, optional (default: None)
        Add this much space between contact map
        and secondary structure. If None, defaults
        to the width of secondary structure * 3.
    """
    def _extract_secstruct(secstruct, axis_range):
        # turn into dictionary representation if
        # passed as a DataFrame
        if isinstance(secstruct, pd.DataFrame):
            # first check we actually have a secondary
            # structure column
            if "sec_struct_3state" not in secstruct.columns:
                return None, None, None

            # do not store any undefined secondary
            # structure in dictionary, or NaN
            # values will lead to problems

            secstruct = secstruct.dropna(
                subset=["sec_struct_3state"]
            )

            secstruct = dict(
                zip(
                    secstruct.id.astype(int),
                    secstruct.sec_struct_3state
                )
            )

        # catch case where there is no secondary
        # structure at all (e.g. because of dataframe
        # full of NaNs)
        if len(secstruct) == 0:
            return None, None, None

        # make sure we only retain secondary structure
        # inside the range of the plot, otherwise
        # drawing artifacts occur
        range_min = min(axis_range)
        range_max = max(axis_range)
        secstruct = {
            i: sstr for (i, sstr) in secstruct.items()
            if range_min <= i < range_max
        }

        first_pos, last_pos = min(secstruct), max(secstruct) + 1
        secstruct_str = "".join(
            [secstruct.get(i, "-") for i in range(first_pos, last_pos)]
        )

        start, end, segments = find_secondary_structure_segments(
            secstruct_str, offset=first_pos
        )

        return start, end, segments

    if ax is None:
        ax = plt.gca()

    if style is None:
        style = {}

    # make secondary structure symmetric if not given
    if secstruct_j is None:
        secstruct_j = secstruct_i

    # get axis ranges to place secondary structure drawings
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    # i corresponds to x-axis, j to y-axis
    if margin is None:
        margin = 3 * style.get("width", 1)
    else:
        margin += style.get("width", 1)

    start_i, end_i, segments_i = _extract_secstruct(secstruct_i, x_range)
    if segments_i is not None:
        secondary_structure_cartoon(
            segments_i,
            **{
                **style,
                "center": max(y_range) + margin,
                "ax": ax,
                "sequence_start": start_i,
                "sequence_end": end_i,
                "horizontal": True,
            }
        )

    start_j, end_j, segments_j = _extract_secstruct(secstruct_j, y_range)
    if segments_j is not None:
        secondary_structure_cartoon(
            segments_j,
            **{
                **style,
                "center": max(x_range) + margin,
                "ax": ax,
                "sequence_start": start_j,
                "sequence_end": end_j,
                "horizontal": False,
            }
        )


def secondary_structure_cartoon(
        sse, ax=None, sequence_start=0,
        sequence_end=None, horizontal=True,
        flip_direction=False, center=0, width=1,
        helix_turn_length=1, strand_width_factor=0.5,
        line_width=2, min_sse_length=0, clipping=False,
        helix_color="k", strand_color="k", coil_color="k",
        draw_coils=True
):
    """
    Plot a 1D secondary structure cartoon.

    Parameters
    ----------
    sse : list
        Secondary structure elements, as returned by
        find_secondary_structure_segments(). Tuples
        in the list have the form (type, start, end),
        where type is either H (helix), E (sheet),
        or C (coil/other)
    ax: matplotlib Axes object
        Axis to draw cartoon on
    sequence_start : float, optional (default:0)
        Plot coordinate of first position from which
        cartoon will be drawn
    sequence_end : float, optional (default: None)
        Last coordinate up to which cartoon will
        be drawn
    horizontal : bool, optional (default: True)
        If True, draw cartoon horizontally, vertically
        otherwise
    flip_direction : bool, optional (default: False)
        Invert drawing direction (from right to left)
    center : float, optional (default: True)
        Center plot coordinate along which cartoon
        will be drawn
    width : float, optional (default: 1)
        Width of secondary structure cartoon
    helix_turn_length : float, optional (default: 1)
        Length for a full helix turn in plot coordinates
        (number of residues per sine)
    strand_width_factor : float, optional (default: 0.5)
        Width of strand relative to full width of cartoon
    line_width : float, optional (default: 2)
        Line width for drawing
    min_sse_length : float, optional (default: 0)
        Only draw secondary structure elements with
        length greater than this threshold
    clipping : bool, optional (default: False)
        Clip drawing at plot axis (must be False
        to draw outside contact map area)
    helix_color : str or tuple, optional (default: "k")
        Matplotlib color to be used for helices
    strand_color : str or tuple, optional (default: "k")
        Matplotlib color to be used for beta strands
    coil_color : str or tuple, optional (default: "k")
        Matplotlib color to be used for all other positions
    draw_coils : bool, optional (Default: True)
        If true, draw line for coil segments.
    """
    def _transform(x, y):
        """
        Transform raw drawing coordinates if
        axis or direction is flipped
        """
        x, y = np.array(x), np.array(y)
        x = -x if flip_direction else x
        return (x, y) if horizontal else (y, x)

    if ax is None:
        ax = plt.gca()

    # collect segments that are neither helix nor
    # strand - this is important since ends of
    # helix segments in plot are not exactly known
    # a priori
    no_ss_segments = [sequence_start]

    # draw secondary structure segments one by one
    for (ss_type, start, end) in sse:
        # do not draw very short elements if chosen
        if end - start < min_sse_length:
            continue

        if ss_type == "H":  # alpha helix
            # length of segment (do not add 1
            # since the number of intervals is
            # one less than the number of residues)
            length = end - start

            # unrounded number of helix turns
            turns = length / float(helix_turn_length)

            # round up to number of complete turns
            # (otherwise helix ends not aligned);
            # make sure there is at least one turn,
            # or code will crash
            full_turns = max(1.0, np.ceil(turns))

            # mismatch between residues covered by rounded
            # number of turns and how many turns the segments
            # actually covers
            overhang = full_turns - turns

            x_sin = np.arange(round(full_turns), step=0.02)
            y = np.sin(2 * np.pi * x_sin) * width + center
            x = start + (x_sin - overhang / 2.0) * helix_turn_length
            ax.plot(
                *_transform(x, y), color=helix_color,
                ls="-", lw=line_width, clip_on=clipping
            )

            # store beginning and end coordinates
            # of helix to fill coil in between

            no_ss_segments.append(x.min())
            no_ss_segments.append(x.max())

        elif ss_type == "E":  # beta strand
            # rectangle part
            if abs(end - start) > 1:
                x_t, y_t = _transform(
                    [start, end - width],
                    [center - width * strand_width_factor,
                     center + width * strand_width_factor]
                )
                ax.add_patch(
                    plt.Rectangle(
                        (x_t[0], y_t[0]),
                        x_t[1] - x_t[0], y_t[1] - y_t[0],
                        edgecolor=strand_color,
                        facecolor=strand_color,
                        clip_on=clipping
                    )
                )

            # triangle part
            x = [end - width, end - width, end]
            y = [center - width, center + width, center]
            ax.add_patch(
                plt.Polygon(
                    list(zip(*_transform(x, y))),
                    edgecolor=strand_color, facecolor=strand_color,
                    clip_on=clipping
                )
            )

            # store beginning and end coordinates
            # of strand to fill coil in between
            no_ss_segments.append(start)
            no_ss_segments.append(end)

        elif ss_type == "-":  # skip drawing (no data)
            no_ss_segments.append(start)
            no_ss_segments.append(end)

    # draw coil until given endpoint
    if sequence_end is not None:
        no_ss_segments.append(sequence_end + 1)

    # finally, draw all coil segments
    if draw_coils:
        for (start, end) in zip(
            no_ss_segments[::2],
            no_ss_segments[1::2]
        ):
            if start > end:
                continue

            x = [start, end]
            y = [center, center]
            ax.plot(*_transform(x, y), color=coil_color,
                    ls="-", lw=line_width, clip_on=clipping)


def find_secondary_structure_segments(sse_string, offset=0):
    """
    Identify segments of secondary structure elements in string

    Parameters
    ----------
    sse_string : str
        String with secondary structure states of sequence
        ("H", "E", "-"/"C")
    offset : int, optional (default: 0)
        Shift start/end indices of segments by this offset

    Returns
    -------
    start : int
        Index of first position (equal to "offset")
    end : int
        Index of last position
    segments : list
        List of tuples with the following elements:

        1. secondary structure element (str)
        2. start position of segment (int)
        3. end position of segment, exlusive (int)
    """
    if len(sse_string) < 1:
        raise ValueError("Secondary structure string must have length > 0.")

    end = len(sse_string) - 1

    sse_list = list(sse_string)
    change_points = [
        (i, (c1, c2)) for (i, (c1, c2)) in
        enumerate(zip(sse_list[:-1], sse_list[1:]))
        if c1 != c2
    ]

    segments = []
    last_start = 0
    # set s2 for the case of only one continuous segment
    s2 = sse_string[0]
    for (p, (s1, s2)) in change_points:
        segments.append((s1, offset + last_start, offset + p + 1))
        last_start = p + 1
    segments.append((s2, offset + last_start, offset + end + 1))

    return offset, end + offset, segments


def ec_lines_pymol_script(ec_table, output_file, distance_cutoff=5,
                          score_column="cn", chain=None):
    """
    Create a Pymol .pml script to visualize ECs on a 3D
    structure
    
    Parameters
    ----------
    ec_table : pandas.DataFrame
        Visualize all EC pairs (columns i, j) in this
        table. If a column "dist" exists and distance_cutoff
        is defined, will assign different colors based on
        the 3D distance of the EC.
    output_file : str
        File path where to store pml script
    distance_cutoff : float, optional (default: 5)
        Color ECs with distance above this threshold
        as false positives (only possible if a column
        "dist" exists in ec_table). If None, will
        use one color for all ECs.
    score_column : str, optional (default: "cn")
        Use this column in ec_table to adjust radius
        of lines. If None, all lines will be drawn
        at equal radius.
    chain : str, optional (default: None)
        Use this PDB chain in residue selection
    """
    t = ec_table.copy()

    # assign line styles
    for prop, val in [
        ("dash_radius", 0.345), ("dash_gap", 0.075), ("dash_length", 0.925)
    ]:
        t.loc[:, prop] = val

    # adjust line width/radius based on score, if selected
    if score_column is not None:
        scaling_factor = 0.5 / ec_table.loc[:, score_column].max()
        t.loc[:, "dash_radius"] = ec_table.loc[:, score_column] * scaling_factor

    if "dist" in ec_table and distance_cutoff is not None:
        t.loc[t.dist <= distance_cutoff, "color"] = "green"
        t.loc[t.dist > distance_cutoff, "color"] = "red"
    else:
        t.loc[:, "color"] = "green"

    if chain is not None:
        chain_sel = ", chain '{}'".format(chain)
    else:
        chain_sel = ""

    with open(output_file, "w") as f:
        f.write("as cartoon{}\n".format(chain_sel))
        f.write("color grey80{}\n".format(chain_sel))
        pymol_pair_lines(t, f, chain)


def enrichment_pymol_script(enrichment_table, output_file,
                            sphere_view=True, chain=None):
    """
    Create a Pymol .pml script to visualize EC "enrichment"

    Parameters
    ----------
    enrichment_table : pandas.DataFrame
        Mapping of position (column i) to EC enrichment
        (column enrichemnt), as returned by 
        evcouplings.couplings.pairs.enrichment()
    output_file : str
        File path where to store pml script
    sphere_view : bool, optional (default: True)
        If True, create pml that highlights enriched positions
        with spheres and color; if False, create pml
        that highlights enrichment using b-factor and
        "cartoon putty"
    chain : str, optional (default: None)
        Use this PDB chain in residue selection
    """
    t = enrichment_table.query("enrichment > 1")

    # compute boundaries for highly coupled residues
    # that will be specially highlighted
    boundary1 = int(0.05 * len(t))  # top 5%
    boundary2 = int(0.15 * len(t))  # top 15%

    t.loc[:, "b_factor"] = t.enrichment

    # set color for "low" enrichment (anything > 1)
    t.loc[:, "color"] = "yelloworange"

    # high
    t.loc[t.iloc[0:boundary1].index, "color"] = "red"

    # medium
    t.loc[t.iloc[boundary1:boundary2].index, "color"] = "orange"

    if sphere_view:
        t.loc[t.iloc[0:boundary2].index, "show"] = "spheres"

    if chain is not None:
        chain_sel = ", chain '{}'".format(chain)
    else:
        chain_sel = ""

    with open(output_file, "w") as f:
        f.write("as cartoon{}\n".format(chain_sel))
        f.write("color grey80{}\n".format(chain_sel))

        if chain is None:
            f.write("alter all, b=0.0\n")
        else:
            f.write("alter chain '{}', b=0.0\n".format(chain))

        pymol_mapping(t, f, chain)

        if not sphere_view:
            f.write("cartoon putty{}\n".format(chain_sel))
