"""
Functions for detecting ECs that should not be
included in 3D structure prediction

Most functions in this module are rewritten from
older pipeline code in choose_CNS_constraint_set.m

Authors:
  Thomas A. Hopf
"""

from operator import xor
from copy import deepcopy


def detect_secstruct_clash(i, j, secstruct):
    """
    Detect if an EC pair (i, j) is geometrically
    impossible given a predicted secondary structure

    Based on direct port of the logic implemented in
    choose_CNS_constraint_set.m from original pipeline,
    lines 351-407.

    Use secstruct_clashes() to annotate an entire
    table of ECs.

    Parameters
    ----------
    i : int
        Index of first position
    j : int
        Index of second position
    secstruct : dict
        Mapping from position (int) to secondary
        structure ("H", "E", "C")

    Returns
    -------
    clashes : bool
        True if (i, j) clashes with secondary
        structure
    """

    # extract a secondary structure substring
    # start and end are inclusive
    def _get_range(start, end):
        return "".join(
            [secstruct[pos] for pos in range(start, end + 1)]
        )

    def _all_equal(string, char):
        return string == len(string) * char

    # get bigger and smaller of the two positions
    b = max(i, j)
    s = min(i, j)

    # if pair too distant in primary sequence, do
    # not consider for clash
    if b - s >= 15:
        return False

    # get secondary structure in range between pairs
    secstruct_string = _get_range(s, b)

    # part 1: check for clashes based on alpha helices
    # first check for helix between them, or both in a helix
    # (or either one directly next to helix)
    if _all_equal(_get_range(s + 1, b - 1), "H"):
        return True
    # of if just one of them is in a helix
    elif xor(secstruct[s] == "H", secstruct[b] == "H"):
        h2 = "H" * (b - s - 1)
        h3 = "H" * (b - s - 2)
        if h2 in secstruct_string:
            if b - s > 6:
                return True
        elif h3 in secstruct_string:
            if b - s > 11:
                return True

    # part 2: check for clashes based on beta strands
    if _all_equal(_get_range(s + 1, b - 1), "E"):
        return True
    elif _all_equal(_get_range(s + 2, b - 2), "E"):
        if b - s > 8:
            return True

    if xor(secstruct[s] == "E", secstruct[b] == "E"):
        e2 = "E" * (b - s - 1)
        e3 = "E" * (b - s - 2)
        e4 = "E" * (b - s - 3)

        if e2 in secstruct_string:
            return True
        elif e3 in secstruct_string:
            return True
        elif e4 in secstruct_string:
            if b - s > 8:
                return True

    return False


def secstruct_clashes(ec_pairs, residues, output_column="ss_clash",
                      secstruct_column="sec_struct_3state"):
    """
    Add secondary structure clashes to EC table

    Parameters
    ----------
    ec_pairs : pandas.DataFrame
        Table with EC pairs that will be tested
        for clashes with secondary structure
        (with columns i, j)
    residues : pandas.DataFrame
        Table with residues in sequence and their
        secondary structure (columns i, ss_pred).
    output_column : str, optional (default: "secstruct_clash")
        Target column indicating if pair is in a
        clash or not
    secstruct_column : str, optional (default: "sec_struct_3state")
        Source column in ec_pairs with secondary structure
        states (H, E, C)

    Returns
    -------
    pandas.DataFrame
        Annotated EC table with clashes
    """
    ec_pairs = deepcopy(ec_pairs)
    secstruct = dict(zip(residues.i, residues[secstruct_column]))

    ec_pairs.loc[:, output_column] = [
        detect_secstruct_clash(row["i"], row["j"], secstruct)
        for idx, row in ec_pairs.iterrows()
    ]

    return ec_pairs


def disulfide_clashes(ec_pairs, output_column="cys_clash"):
    """
    Add disulfide bridge clashes to EC table (i.e. if
    any cysteine residue is coupled to another cysteine).
    This flag is necessary if disulfide bridges are created
    during folding, since only one bridge is possible per
    cysteine.

    Parameters
    ----------
    ec_pairs : pandas.DataFrame
        Table with EC pairs that will be tested
        for the occurrence of multiple cys-cys
        pairings (with columns i, j, A_i, A_j)
    output_column : str, optional (default: "cys_clash")
        Target column indicating if pair is in a
        clash or not

    Returns
    -------
    pandas.DataFrame
        Annotated EC table with clashes
    """
    ec_pairs = deepcopy(ec_pairs)

    # find all cys-cys pairs
    cys_pairs = ec_pairs.query("A_i == 'C' and A_j == 'C'")

    # detect multiple occurrences of cysteine residues in
    # cys-cys bridges

    paired = set()
    clashes = []

    # go through all cys-cys ECs
    for idx, row in cys_pairs.iterrows():
        i, j = row["i"], row["j"]
        # have we seen either residue as paired before?
        # if so, flag as a clash
        if i in paired or j in paired:
            clashes.append(idx)

        # store that we have seen both residues in
        # a pair
        paired.add(i)
        paired.add(j)

    # initialize output to no clash for all
    ec_pairs.loc[:, output_column] = False

    # then set clash flag for detected clashes
    ec_pairs.loc[clashes, output_column] = True

    return ec_pairs
