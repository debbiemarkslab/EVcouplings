"""
Visualization of evolutionary couplings (contact maps etc.)

Authors:
  Thomas A. Hopf
"""

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_contact_map(ecs=None, monomer=None, multimer=None,
                     distance_cutoff=5, ax=None, scale_sizes=True,
                     ec_style=STYLE_EC, monomer_style=STYLE_CONTACT,
                     multimer_style=STYLE_CONTACT_MULTIMER,
                     margin=5, invert_y=True, bound_by_structure=True,
                     show_secstruct=True):
    """
    Wrapper for simple contact map plots with monomer and
    multimer contacts. For full flexibility, compose your own
    contact map plot using the functions used below.

    Parameters
    ----------
    # TODO
    """
    if ecs is None and monomer is None and multimer is None:
        raise ValueError(
            "Need to specify at least one of ecs, monomer or multimer"
        )

    if ax is None:
        ax = plt.gca()

    # determine and set plot boundaries
    if bound_by_structure:
        if monomer is None:
            raise ValueError(
                "Cannot determine plot boundaries from structure "
                "since no monomer distance map is given"
            )
        bounding_data = monomer.contacts(distance_cutoff)

    else:
        if ecs is None:
            raise ValueError(
                "Cannot determine plot boundaries from ECs "
                "since no EC list is given"
            )

        bounding_data = ecs

    set_range(bounding_data, ax=ax, margin=margin, invert_y=invert_y)

    # enable rescaling of points and secondary structure if necessary
    if scale_sizes:
        scale_func = lambda x: scale(x, ax=ax)
    else:        
        scale_func = lambda x: x

    # plot monomer contacts
    if monomer is not None:
        plot_pairs(
            monomer.contacts(distance_cutoff),
            symmetric=False, style=scale_func(STYLE_CONTACT)
        )

    # plot multimer contacts
    if multimer is not None:
        plot_pairs(
            multimer.contacts(distance_cutoff),
            symmetric=False, style=scale_func(STYLE_CONTACT_MULTIMER)
        )

    # plot ECs
    if ecs is not None:
        plot_pairs(ecs, symmetric=True, style=scale_func(STYLE_EC))

    # plot secondary structure
    if show_secstruct:
        try:
            plot_secondary_structure(
                monomer.residues_i, monomer.residues_j, style=scale_func(STYLE_SECSTRUCT)
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


def set_range(pairs, symmetric=True, ax=None, margin=0, invert_y=True):
    """
    Set axes ranges for contact map based
    on minimal/maximal values in pair list

    Parameters
    ----------
    pairs : pandas.DataFrame
        DataFrame with pairs (will be extracted
        from columns named "i" and "j" and
        converted to integer values and used
        to define x- and y-axes, respectively)
    symmetric : bool, optional (default:True)
        If true, will define range on joint set
        of values in columns i and j, resulting
        in a square contact map
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
    """
    i = pairs.i.astype(int)
    j = pairs.j.astype(int)

    if ax is None:
        ax = plt.gca()

    if symmetric:
        x_range = (
            min(i.min(), j.min()) - margin,
            max(i.max(), j.max()) + margin
        )
        y_range = x_range
    else:
        x_range = (i.min() - margin, i.max() + margin)
        y_range = (j.min() - margin, j.max() + margin)

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

    L = min(
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
        secondary structure character ("H", "E", "-"/"C"),
        or a DataFrame with columns "id" and "sec_struct_3state"
        (as returned by DistanceMap.residues_i
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
    def _extract_secstruct(secstruct):
        # turn into dictionary representation if
        # passed as a DataFrame
        if isinstance(secstruct, pd.DataFrame):
            secstruct = dict(
                zip(
                    secstruct.id.astype(int),
                    secstruct.sec_struct_3state
                )
            )

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

    start_i, end_i, segments_i = _extract_secstruct(secstruct_i)
    start_j, end_j, segments_j = _extract_secstruct(secstruct_j)

    # get axis ranges to place secondary structure drawings
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    # i corresponds to x-axis, j to y-axis
    if margin is None:
        margin = 3 * style.get("width", 1)
    else:
        margin += style.get("width", 1)

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

    ax: matplotlib axis to draw on

    sse: list of secondary structure elements:
         (type, start, end), where type is either
         H (helix), E (sheet), or C (coil/other)

    sequence_start: index of first residue in sequence
    horizontal: plot secondary structure horizontally or
         vertically

    flip_direction: go from right to left / top to bottom
    center: coordinate along which 1D plot runs
    width: width of plot around center
    helix_turn_length: number of residues per sine
    strand_width_factor: extension of strand rectangle around
        center as a fraction of width
    line_width: line width (points) for sequence/helix plotting
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

    End index is exclusive.
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
