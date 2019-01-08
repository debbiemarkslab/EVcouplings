"""
Functions for generating input to Haddock webserver

Authors:
  Anna G. Green
"""


def haddock_dist_restraint(resid_i, chain_i, resid_j, chain_j,
                           dist, lower, upper, atom_i=None, atom_j=None,
                           comment=None):
    """
    Create an ambiguous restraint string for uploading into
    Haddock v2.2 webserver.

    Parameters
    ----------
    resid_i : int
        Index of first residue
    chain_i : str
        Name of first chain (interpreted as segid for docking)
    resid_j : int
        Index of second residue
    chain_j : str
        Name of second chain (interpreted as segid for docking)
    dist : float
        Restrain distance between residues to this value
    lower : float
        Lower bound delta on distance
    upper : float
        Upper bound delta on distance
    atom_i : str, optional (default: None)
        Name of first atom
    atom_j : str, optional (default: None)
        Name of first atom
    comment : str, optional (default: None)
        Print comment at the end of restraint line

    Returns
    -------
    r : str
        Distance restraint
    """

    if comment is not None:
        comment_str = "{}".format(comment)
    else:
        comment_str = ""

    if atom_i is not None:
        # TODO: adding the atom strings causes an error in docking. Needs debugging.
        # atom_str_i = " and name {}".format(atom_i)
        atom_str_i = ""
    else:
        atom_str_i = ""

    if atom_j is not None:
        # atom_str_j = " and name {}".format(atom_j)
        atom_str_j = ""
    else:
        atom_str_j = ""

    r = (
        "! {}\n"
        "assign (resid {} and segid {}{})\n"
        "(\n"
        " (resid {} and segid {}{})\n"
        ") {} {} {}".format(
            comment_str,
            resid_i, chain_i, atom_str_i,
            resid_j, chain_j, atom_str_j,
            dist, upper, lower
        )
    )

    return r
