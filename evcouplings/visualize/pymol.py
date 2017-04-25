"""
Visualization of properties on 3D structures using
Pymol pml scripts

Authors:
  Thomas A. Hopf
"""


def pymol_secondary_structure(residues, output_file, chain=None,
                              sec_struct_column="sec_struct_3state"):
    """
    Assign predicted secondary structure to 3D structure in Pymol
    
    Parameters
    ----------
    residues : pandas.DataFrame
        Table with secondary structure prediction/assignment
        (position in column i)
    output_file : str
        Write Pymol script to this file
    chain : str, optional (default: None)
        PDB chain that should be targeted by secondary
        structure assignment. If None, residues will be selected
        py position alone, which may cause wrong assignments
        if multiple chains are present in the structure.
    sec_struct_column : str, optional (default: sec_struct_3state)
        Column in residues table that contains the secondary 
        structure symbols for each position (convention: helix: H,
        strand: E, other: C)
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

    with open(output_file, "w") as f:
        f.write("\n".join(cmds))


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
        (prefixed with 0x instead of #).
    output_file : str or file-like
        Write Pymol script to this file (filename as string,
        or writeable file handle)
    chain : str or dict(str -> str), optional (default: None)
        PDB chain(s) that should be targeted by line drawing
        - If None, residues will be selected
          py position alone, which may cause wrong assignments
          if multiple chains are present in the structure.
        - Different chains can be assigned for each i and j, 
          if a dictionary that maps from segment (str) to PDB chain (str)
          is given. In this case, columns "segment_i" and "segment_j"
          must be present in the pairs dataframe.
    atom : str, optional (default: "CA")
        Put end of line on this atom in each residue
    """
    cmds = []

    def _selector(row, column):
        if chain is not None:
            # if we have a dictionary, map chain based on selector column
            if isinstance(chain, dict):
                c = chain[row["segment_" + column]]
            else:
                # otherwise just take the name of the chain as it is
                c = chain

            # create selector for chain
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
        if "color" in r:
            cmds.append("color {}, {}".format(r["color"], id_))

        # check all other properties we can set using
        # set <parameter>, <value>, <object>
        for param in ["dash_radius", "dash_gap", "dash_length"]:
            if param in r:
                cmds.append(
                    "set {}, {}, {}".format(param, r[param], id_)
                )

    cmd_str = "\n".join(cmds)

    try:
        output_file.write(cmd_str)
    except AttributeError:
        with open(output_file, "w") as f:
            f.write(cmd_str)

    return cmds
