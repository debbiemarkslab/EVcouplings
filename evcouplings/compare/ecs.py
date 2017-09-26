"""
Compare evolutionary couplings to distances in 3D structures

Authors:
  Thomas A. Hopf
"""

import numpy as np


def add_distances(ec_table, dist_map, target_column="dist"):
    """
    Add pair distances to EC score table

    Parameters
    ----------
    ec_table : pandas.DataFrame
        List of evolutionary couplings, with pair
        positions in columns i and j
    dist_map : DistanceMap
        Distance map that will be used to annotate
        distances in ec_table
    target_column : str
        Name of column in which distances will be stored

    Returns
    -------
    pandas.DataFrame
        Couplings table with added distances
        in target_column. Pairs where no distance
        information is available will be np.nan
    """
    ec_table = ec_table.copy()

    ec_table.loc[:, target_column] = [
        dist_map.dist(i, j, raise_na=False)
        for i, j in zip(ec_table.i, ec_table.j)
    ]

    return ec_table


def add_precision(ec_table, dist_cutoff=5, score="cn",
                  min_sequence_dist=6, target_column="precision",
                  dist_column="dist"):
    """
    Compute precision of evolutionary couplings as predictor
    of 3D structure contacts

    Parameters
    ----------
    ec_table : pandas.DataFrame
        List of evolutionary couplings
    dist_cutoff : float, optional (default: 5)
        Upper distance cutoff (in Angstrom) for a
        pair to be considered a true positive contact
    score : str, optional (default: "cn")
        Column which contains coupling score. Table will
        be sorted in descending order by this score.
    min_sequence_dist : int, optional (default: 6)
        Minimal distance in primary sequence for an EC to
        be included in precision calculation
    target_column : str, optional (default: "precision")
        Name of column in which precision will be stored
    dist_column : str, optional (default: "dist")
        Name of column which contains pair distances

    Returns
    -------
    pandas.DataFrame
        EC table with added precision values as a
        function of EC rank (returned table will be
        sorted by score column)
    """
    # make sure list is sorted by score
    ec_table = ec_table.sort_values(by=score, ascending=False)

    if min_sequence_dist is not None:
        ec_table = ec_table.query("abs(i - j) >= @min_sequence_dist")

    ec_table = ec_table.copy()

    # number of true positive contacts
    true_pos_count = (ec_table.loc[:, dist_column] <= dist_cutoff).cumsum()

    # total number of contacts with specified distance
    pos_count = ec_table.loc[:, dist_column].notnull().cumsum()

    ec_table.loc[:, target_column] = true_pos_count / pos_count
    return ec_table


def coupling_scores_compared(ec_table, dist_map, dist_map_multimer=None,
                             dist_cutoff=5, output_file=None, score="cn",
                             min_sequence_dist=6):
    """
    Utility function to create "CouplingScores.csv"-style
    table

    Parameters
    ----------
    ec_table : pandas.DataFrame
        List of evolutionary couplings
    dist_map : DistanceMap
        Distance map that will be used to annotate
        distances in ec_table
    dist_map_multimer : DistanceMap, optional (default: None)
        Additional multimer distance map. If given,
        the distance for any EC pair will be the minimum
        out of the monomer and multimer distances.
    dist_cutoff : float, optional (default: 5)
        Upper distance cutoff (in Angstrom) for a
        pair to be considered a true positive contact
    output_file : str, optional (default: None)
        Store final table to this file
    score : str, optional (default: "cn")
        Column which contains coupling score. Table will
        be sorted in descending order by this score.
    min_sequence_dist : int, optional (default: 6)
        Minimal distance in primary sequence for an EC to
        be included in precision calculation

    Returns
    -------
    pandas.DataFrame
        EC table with added distances, and precision
        if dist_cutoff is given.
    """
    if dist_map_multimer is None:
        x = add_distances(ec_table, dist_map)
    else:
        x = add_distances(ec_table, dist_map, "dist_intra")
        x = add_distances(x, dist_map_multimer, "dist_multimer")
        x.loc[:, "dist"] = np.fmin(
            x.dist_intra, x.dist_multimer
        )

    if min_sequence_dist is not None:
        x = x.query("abs(i - j) >= @min_sequence_dist")

    # if distance cutoff is given, add precision
    if dist_cutoff is not None:
        x = add_precision(
            x, dist_cutoff, score=score,
            min_sequence_dist=min_sequence_dist
        )

    # also save to file if path is given
    if output_file is not None:
        x.to_csv(output_file, index=False)

    return x
