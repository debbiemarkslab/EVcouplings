"""
Functions for generating input to Haddock webserver

Authors:
  Anna G. Green
"""


def haddock_dist_restraint(resid_i, chain_i, resid_j, chain_j,
                           dist, lower, upper, comment=None):
    """
    Create a CNS distance restraint string

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
    comment : str, optional (default: None)
        Print comment at the end of restraint line

    Returns
    -------
    r : str
        Distance restraint
    """

    if comment is not None:
        comment_str = "! {}".format(comment)
    else:
        comment_str = ""

    r = (
        "{}\n"
        "assign (resid {} and segid {})\n"
        "(\n"
        " (resid {} and name {}) \n"
        ") {} {} {}".format(
            comment_str,
            resid_i, chain_i,
            resid_j, chain_j,
            dist, upper, lower
        )
    )

    return r
