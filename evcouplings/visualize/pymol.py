"""
Visualization of properties on 3D structures using
Pymol pml scripts

Authors:
  Thomas A. Hopf
"""

import pandas as pd


def _write_pymol_commands(commands, output_file):
    """
    Helper function to write a set of lines to a file

    Parameters
    ----------
    commands : list(str)
        List of commands, each will be written to a new
        line.
    output_file : str or file-like:
        Path to output file (str), or writeable
        file handle, in which commands will be stored
    """
    cmd_str = "\n".join(commands) + "\n"

    try:
        output_file.write(cmd_str)
    except AttributeError:
        with open(output_file, "w") as f:
            f.write(cmd_str)


def pymol_secondary_structure(residues, output_file, chain=None,
                              sec_struct_column="sec_struct_3state"):
    """
    Assign predicted secondary structure to 3D structure in Pymol

    Parameters
    ----------
    residues : pandas.DataFrame
        Table with secondary structure prediction/assignment
        (position in column i)
    output_file : str or file-like
        Write Pymol script to this file (filename as string,
        or writeable file handle)
    chain : str, optional (default: None)
        PDB chain that should be targeted by secondary
        structure assignment. If None, residues will be selected
        py position alone, which may cause wrong assignments
        if multiple chains are present in the structure.
    sec_struct_column : str, optional (default: sec_struct_3state)
        Column in residues table that contains the secondary
        structure symbols for each position (convention: helix: H,
        strand: E, other: C)

    Returns
    -------
    cmds : list
        List of generated pymol commands (written to output_file)
    """
    cmds = []

    # define chain selector if chain is given
    if chain is not None:
        chain_sel = " and chain '{}'".format(chain)
    else:
        chain_sel = ""

    # mapping from our secondary structure state
    # convention to what Pymol expects
    state_mapping = {
        "H": "H",
        "E": "S",
    }

    # go through all residues
    for idx, r in residues.iterrows():
        sec_struct = r[sec_struct_column]

        # see if we have helix or sheet prediction
        if sec_struct in state_mapping:
            # create pymol command
            cmd = "alter (resi {}{}), ss='{}'".format(
                r["i"], chain_sel, state_mapping[sec_struct]
            )
            cmds.append(cmd)

    # at the end we also need to issue a rebuild so
    # the changes are displayed
    cmds.append("rebuild")

    _write_pymol_commands(cmds, output_file)
    return cmds


def pymol_pair_lines(pairs, output_file, chain=None, atom="CA", pair_prefix="ec"):
    """
    Draw lines between residue pairs on a structure using Pymol

    Parameters
    ----------
    pairs : pandas.DataFrame
        Table of pairs to draw (positions in columns i and j).
        If this dataframe contains columns named "color",
        "dash_radius", "dash_gap" or "dash_length", these will
        be used to adjust the corresponding Pymol property of each
        drawn line. Dash parameters should be floats, colors can
        be any of the Pymol color names or a hexademical color code
        (# will be converted to Pymol 0x convention automatically).
        If columns "chain_i" or "chain_j" are present, these will
        define the respective chains for each end of the line, and
        override the chain parameter.
    output_file : str or file-like
        Write Pymol script to this file (filename as string,
        or writeable file handle)
    chain : str or dict(str -> str), optional (default: None)
        PDB chain(s) that should be targeted by line drawing

        * If None, residues will be selected
          py position alone, which may cause wrong assignments
          if multiple chains are present in the structure.

        * Different chains can be assigned for each i and j,
          if a dictionary that maps from segment (str) to PDB chain (str)
          is given. In this case, columns "segment_i" and "segment_j"
          must be present in the pairs dataframe.

    atom : str, optional (default: "CA")
        Put end of line on this atom in each residue

    Returns
    -------
    cmds : list
        List of generated pymol commands (written to output_file)
    """
    cmds = []

    def _selector(row, column):
        # if there is a chain specified, take it
        if "chain_" + column in row:
            c = row["chain_" + column]
        elif chain is not None:
            # if we have a dictionary and a segment column,
            # use that
            if isinstance(chain, dict):
                c = chain[row["segment_" + column]]
            else:
                # otherwise just take the name of the chain as it is
                c = chain
        else:
            c = None

        # create selector for chain
        if c is not None:
            chain_sel = "chain '{}' and ".format(c)
        else:
            # otherwise, if no chain info given, do not put selector on chain
            chain_sel = ""

        # return full residue selector
        return "{}resid {} and name {}".format(
            chain_sel, r[column], atom
        )

    # go through all pairs
    for i, (idx, r) in enumerate(pairs.iterrows(), start=1):
        sel_i = _selector(r, "i")
        sel_j = _selector(r, "j")
        id_ = "{}{}".format(pair_prefix, i)

        cmd = "dist {}, {}, {}, label=0".format(
            id_, sel_i, sel_j
        )
        cmds.append(cmd)

        # check if we need to set color
        if "color" in r and pd.notnull(r["color"]):
            cmds.append(
                "color {}, {}".format(
                    r["color"].replace("#", "0x"),
                    id_
                )
            )

        # check all other properties we can set using
        # set <parameter>, <value>, <object>
        for param in ["dash_radius", "dash_gap", "dash_length"]:
            if param in r and pd.notnull(r[param]):
                cmds.append(
                    "set {}, {}, {}".format(param, r[param], id_)
                )

    _write_pymol_commands(cmds, output_file)
    return cmds


def pymol_mapping(mapping, output_file, chain=None, atom=None):
    """
    Map properties onto individual residues of
    structure using Pymol

    Parameters
    ----------
    mapping : pandas.DataFrame
        Table of properties to map on residues.

        * positions in column "i"

        * color in column "color" (hexadecimal
          color codes have to start with a "#",
          and will be converted to Pymol format
          starting with 0x automatically)

        * display type in column "show"

        * b factor in column "b_factor"

    output_file : str or file-like
        Write Pymol script to this file (filename as string,
        or writeable file handle)
    chain : str, optional (default: None)
        PDB chain that should be targeted by secondary
        structure assignment. If None, residues will be selected
        py position alone, which may cause wrong assignments
        if multiple chains are present in the structure.
    atom : str, optional (default: None)
        Only apply mapping to this atom in each residue

    Returns
    -------
    cmds : list
        List of generated pymol commands (written to output_file)
    """
    cmds = []

    # define chain selector if chain is given
    if chain is not None:
        chain_sel = " and chain '{}'".format(chain)
    else:
        chain_sel = ""

    # define atom selector, if given
    if atom is not None:
        atom_sel = " and name {}".format(atom)
    else:
        atom_sel = ""

    # create Pymol commands for each row in mapping table
    for i, (idx, r) in enumerate(mapping.iterrows(), start=1):
        sel = "resid {}{}{}".format(
            r["i"], chain_sel, atom_sel
        )

        # adjust color, display type and b_factor if
        # the respective value is given (i.e., not nan)
        if "color" in r and pd.notnull(r["color"]):
            cmds.append(
                "color {}, {}".format(
                    r["color"].replace("#", "0x"),
                    sel
                )
            )

        if "show" in r and pd.notnull(r["show"]):
            cmds.append(
                "show {}, {}".format(r["show"], sel)
            )

        if "b_factor" in r and pd.notnull(r["b_factor"]):
            cmds.append(
                "alter {}, b={}".format(sel, r["b_factor"])
            )
            pass

    _write_pymol_commands(cmds, output_file)
    return cmds
