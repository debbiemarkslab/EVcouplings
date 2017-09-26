"""
Functions for detecting ECs that should not be
included in 3D structure prediction

Most functions in this module are rewritten from
older pipeline code in choose_CNS_constraint_set.m

Authors:
  Thomas A. Hopf
"""

from collections import defaultdict
from itertools import combinations, product

import pandas as pd
import numpy as np

from evcouplings.compare.pdb import ClassicPDB
from evcouplings.utils.calculations import dihedral_angle
from evcouplings.utils.config import InvalidParameterError
from evcouplings.visualize.pairs import find_secondary_structure_segments


def _alpha_dihedrals(coords, segments):
    """
    Compute dihedral score for alpha-helical
    segments.

    This function reimplements the functionality of
    analyze_alpha_helix.pl and make_alpha_beta_score_table.m
    (analyze_alpha_helix function) from the original pipeline.

    Parameters
    ----------
    coords : pandas.DataFrame
        Coordinate and residue information
        dataframe (residues with CA
        coordinates only)
    segments : list
        List of tuples (start, end) that
        define secondary structure segments
        (end index is exclusive)

    Returns
    -------
    pandas.DataFrame
        Table with dihedral angles for all
        helical residues with defined angle
        (columns: helix (helix index), i (position),
        dihedral (dihedral angle))
    """
    # check if we have cooords for position
    def has(pos):
        return pos in coords.i.values

    # return coords for position
    def xyz(pos):
        return coords.loc[
            coords.i == pos, ["x", "y", "z"]
        ].iloc[0].values

    res = []
    # go through all helical segments
    for helix_idx, (start, end) in enumerate(segments):
        # compute dihedral for each residue
        # within that helical segment
        for i in range(start, end):
            # check if all four necessary CA atoms are defined
            if has(i - 1) and has(i) and has(i + 1) and has(i + 2):
                # compute dihedral
                angle = dihedral_angle(
                    xyz(i - 1), xyz(i), xyz(i + 1), xyz(i + 2)
                )
                res.append((helix_idx, i, angle))

    return pd.DataFrame(
        res, columns=["helix", "i", "dihedral"]
    )


def _beta_dihedrals(coords, segments, max_strand_distance=7,
                    original=True):
    """
    Compute dihedral score for beta strand
    segments.

    This function reimplements the functionality of
    analyze_beta_strand.pl and make_alpha_beta_score_table.m
    (analyze_beta_strand function) from the original pipeline.

    Parameters
    ----------
    coords : pandas.DataFrame
        Coordinate and residue information
        dataframe (residues with CA
        coordinates only)
    segments : list
        List of tuples (start, end) that
        define secondary structure segments
        (end index is exclusive)
    max_strand_distance : float, optional (default: 7)
        Maximum distance of strands to be considered
        for dihedral angle calculation
    original : bool, optional (default: True):
        Use exact implementation of 2011 ranking protocol

    Returns
    -------
    pandas.DataFrame
        Table with dihedral angles for all
        sheet residues with defined angle
        (columns: helix (helix index), i (position),
        dihedral (dihedral angle))
    """
    # transfer coords into a dict of numpy vectors for faster access
    coords = dict(
        zip(coords.i, coords.loc[:, ["x", "y", "z"]].values)
    )

    # check if we have cooords for position
    def has(pos):
        return pos in coords

    # return coords for position
    def xyz(pos):
        return coords[pos]

    # determine orientation of strands (parallel or antiparallel)
    def _orientation(pairs):
        # votes in favor of parallel arrangement, and total votes
        par, total = 0, 0
        for idx, r in pairs.iterrows():
            i, j = r["pos_i"], r["pos_j"]
            if has(i - 2) and has(i + 2) and has(j + 2):
                d_par = np.linalg.norm(xyz(i + 2) - xyz(j + 2))

                if original:
                    # the residues chosen here disagree with residues
                    # for angle calculation
                    d_anti = np.linalg.norm(xyz(i - 2) - xyz(j + 2))
                else:
                    # alternative implementation consistent with angle
                    # calculation
                    d_anti = np.linalg.norm(xyz(i + 2) - xyz(j - 2))

                # cast votes
                total += 1
                if d_par <= d_anti:
                    par += 1

        if total == 0:
            return 0
        else:
            return par / total * 2 - 1

    # compute dihedral angles for residue pairings
    def _compute_dihedral(pairs, strands_parallel):
        res = []
        for idx, r in pairs.iterrows():
            i, j = r["pos_i"], r["pos_j"]
            if original:
                # the residues evaluated here disagree with residues
                # used for angle calculation
                if not has(j + 2):
                    continue

                if strands_parallel and not has(i + 2):
                    continue

                if not strands_parallel and not has(i - 2):
                    continue
            else:
                # alternative implementation consistent with angle
                # calculation
                if not has(i + 2):
                    continue

                if strands_parallel and not has(j + 2):
                    continue

                if not strands_parallel and not has(j - 2):
                    continue

            if strands_parallel:
                angle = dihedral_angle(
                    xyz(i), xyz(i + 2), xyz(j + 2), xyz(j)
                )
            else:
                angle = dihedral_angle(
                    xyz(i), xyz(i + 2), xyz(j - 2), xyz(j)
                )

            res.append(
                (i, j, int(r["strand_i"]), int(r["strand_j"]), angle)
            )

        return pd.DataFrame(
            res, columns=["i", "j", "strand_i", "strand_j", "dihedral"]
        )

    # go through all possible combinations of strands
    # and identify what possible partners are in 3D
    strand_partners = defaultdict(list)

    for (strand_i, seg_i), (strand_j, seg_j) in combinations(enumerate(segments), 2):
        # compute all pairwise C_alpha distances between residues in
        # strands i and j
        pair_dists = pd.DataFrame(
            [
                (
                    strand_i, strand_j, pos_i, pos_j,
                    np.linalg.norm(xyz(pos_i) - xyz(pos_j))
                )
                for (pos_i, pos_j) in product(range(*seg_i), range(*seg_j))
                if has(pos_i) and has(pos_j)
            ], columns=["strand_i", "strand_j", "pos_i", "pos_j", "dist"]
        )
        # test if strands are proximal, otherwise skip pair
        if pair_dists.dist.min() > max_strand_distance:
            continue

        # for each position in strand i,
        # identify nearest residue in strand j
        nearest = pair_dists.sort_values(
            by=["pos_i", "dist"]
        ).groupby(["pos_i"]).first().reset_index()

        # identify pairs that are close according to distance threshold
        close = nearest.loc[nearest.dist <= max_strand_distance]

        # also determine strand distance as closest of all pairs
        strand_dist = close.dist.min()

        # select anything between first and last close pair, this
        # defines the strand segments for further computations including angles
        first_good, last_good = close.index.min(), close.index.max()
        dihedral_pairs = nearest.loc[first_good:last_good]

        # determine if pair is parallel or antiparallel
        orientation_vote = _orientation(dihedral_pairs)
        strands_parallel = orientation_vote > 0

        # compute dihedrals (if we have the positions in coordinates)
        # print("ORIENTATION:", orientation_vote, strands_parallel)
        dihedrals = _compute_dihedral(dihedral_pairs, strands_parallel)

        # store strand pairing
        strand_partners[strand_i].append(
            (strand_dist, strand_j, dihedrals)
        )

    # now assemble final table of dihedrals,
    # allow only up to two pairings per strand
    # (if more, take the two closest ones in 3D based on strand_dist)
    all_dihedrals = pd.DataFrame()
    num_partners = defaultdict(int)
    for strand_i, partners in strand_partners.items():
        # only take two closest strands
        for dist, strand_j, dihedrals in sorted(partners):
            # check we do not have two partners yet
            if num_partners[strand_i] < 2 and num_partners[strand_j] < 2:
                all_dihedrals = all_dihedrals.append(dihedrals)
                num_partners[strand_i] += 1
                num_partners[strand_j] += 1

    return all_dihedrals


def dihedral_ranking_score(structure, residues, sec_struct_column="sec_struct_3state",
                           original=True):
    """
    Assess quality of structure model by twist of 
    predicted alpha-helices and beta-sheets.

    This function re-implements the functionality of
    make_alpha_beta_score_table.m from the original
    pipeline.

    Parameters
    ----------
    structure : evcouplings.compare.pdb.Chain
        Chain with 3D structure coordinates to evaluate
    residues : pandas.DataFrame
        Residue table with secondary structure predictions
        (columns i, A_i and secondary structure column)
    sec_struct_column : str, optional (default: sec_struct_3state)
        Column in residues dataframe that contains predicted
        secondary structure (H, E, C)
    original : bool, optional (default: True):
        Use exact implementation of 2011 ranking protocol

    Returns
    -------
    int
        Number of alpha dihedrals computed
    float
        Alpha dihedral score (unnormalized)
    int
        Number of beta dihedrals computed
    float
        Beta dihedral score (unnormalized)
    """
    # create table that, for each residue, contains
    # secondary structure and C_alpha coordinates.

    # First, throw away anything but C_alpha atoms
    structure = structure.filter_atoms(atom_name="CA")

    # Then, merge residue with atom information
    x = structure.residues.merge(
        structure.coords, left_index=True,
        right_on="residue_index"
    )

    # Then, merge with secondary structure prediction
    # PDB indices are strings, so merge on string
    residues = residues.copy()
    residues.loc[:, "id"] = residues.i.astype(str)
    x = residues.merge(
        x, on="id", how="left", suffixes=("", "_")
    )

    # find secondary structure segments
    _, _, segments = find_secondary_structure_segments(
        "".join(x.loc[:, sec_struct_column]), offset=x.i.min()
    )

    def _get_segments(seg_type):
        return [
            (start, end) for (type_, start, end) in segments
            if type_ == seg_type
        ]

    segs_alpha = _get_segments("H")
    segs_beta = _get_segments("E")

    # extract positions that actually have C_alpha coordinates
    x_valid = x.dropna(subset=["x", "y", "z"])

    # compute alpha helix and beta sheet dihedrals
    d_alpha = _alpha_dihedrals(x_valid, segs_alpha)
    d_beta = _beta_dihedrals(x_valid, segs_beta, original=original)

    # Finally, merge results into score

    # first, count how many angles in the right range
    # we have, but weight for actual value of the dihedral

    # alpha
    alpha_weights = [
        (0.2, 0.44, 0.52),
        (0.4, 0.52, 0.61),
        (0.6, 0.61, 0.70),
        (0.8, 0.70, 0.78),
        (1.0, 0.78, 0.96),
        (0.8, 0.96, 1.05),
        (0.6, 1.05, 1.13),
        (0.4, 1.13, 1.22),
        (0.2, 1.22, 1.31),
    ]

    if len(d_alpha) > 0:
        alpha_dihedral_score = sum([
            weight * len(d_alpha.query(
                "@lower < dihedral and dihedral <= @upper")
            )
            for weight, lower, upper in alpha_weights
        ])
    else:
        alpha_dihedral_score = 0

    # beta
    beta_weights = [
        (0.2, -0.3, -0.1),
        (0.4, -0.4, -0.3),
        (0.6, -0.5, -0.4),
        (0.8, -0.6, -0.5),
        (1.0, -0.8, -0.6),
        (0.8, -0.9, -0.8),
        (0.6, -1.0, -0.9),
        (0.4, -1.1, -1.0),
        (0.2, -1.2, -1.1),
    ]

    if len(d_beta) > 0:
        beta_dihedral_score = sum([
            weight * len(d_beta.query(
                "@lower <= dihedral and dihedral < @upper")
            )
            for weight, lower, upper in beta_weights
        ])
    else:
        beta_dihedral_score = 0

    return len(d_alpha), alpha_dihedral_score, len(d_beta), beta_dihedral_score


def dihedral_ranking(structure_files, residues, chain=None,
                     sec_struct_column="sec_struct_3state",
                     model=0):
    """
    Assess quality of a set of structure models by
    twist of predicted alpha-helices and beta-sheets.

    This function re-implements the final score table
    computed in make_alpha_beta_score_table.m from
    the original pipeline. Some of the implementation
    details where however modified, possibly leading
    to differences in the final computed ranking scores.

    Parameters
    ----------
    structure_files : list(str)
        Paths to PDB files that will be ranked
        (have to be in .pdb format)
    residues : pandas.DataFrame
        Residue table with secondary structure predictions
        (columns i, A_i and secondary structure column)
    chain : str, optional (default: None)
        Use this chain in each structure for the calculation.
        If None, will pick the only existing chain (if there
        are multiple chains, an InvalidParameterError will be
        raised).
    sec_struct_column : str, optional (default: sec_struct_3state)
        Column in residues dataframe that contains predicted
        secondary structure (H, E, C)
    model : int, optional (default: 0)
        Use this model from each PDB structure
        
    Returns
    -------
    pandas.DataFrame
        Table with final ranking scores (column ranking_score)
        and alpha and beta dihedral scores, as well
        as well as possible maximum score (equal to the number
        of computed dihedrals).
        
    Raises
    ------
    InvalidParameterError
        If chain is None but multiple chains exist in any
        of the structures
    """
    res = []
    for filename in structure_files:
        # load structure
        struc = ClassicPDB.from_file(filename)

        # see if we can select the right chain
        if chain is None:
            chains = struc.model_to_chains[model]
            if len(chains) != 1:
                raise InvalidParameterError(
                    "Model has more than one chain, need to "
                    "specify chain parameter to disambiguate."
                )
            chain = chains[0]

        # extract chain from structure
        sel_chain = struc.get_chain(chain)

        # compute dihedral ranking score
        x = dihedral_ranking_score(
            sel_chain, residues, sec_struct_column, original=False
        )
        res.append((filename, *x))

    r = pd.DataFrame(
        res, columns=[
            "filename", "num_alpha_dihedrals", "alpha_dihedral_score",
            "num_beta_dihedrals", "beta_dihedral_score"
        ]
    )

    # maximum score we could have obtained for either set of dihedrals
    max_alpha = r.num_alpha_dihedrals.max()

    # note that, unlike for alpha dihedrals, beta dihedrals are dependent
    # on what contacts between strands are made in 3D structure models
    max_beta = r.num_beta_dihedrals.max()

    # compute final ranking score
    # this computation is somewhat diffeent from original implementation:
    # - normalization for alpha helices is through number of possible dihedrals
    #   not the number of residues with helix secondary structure
    # - scores for helix and beta dihedrals are not adjusted to values < 0 in
    #   borderline cases

    # make sure we do not divide by 0 if we didn't count any dihedrals at all
    max_val = max(1, max_alpha + max_beta)

    r.loc[:, "ranking_score"] = (
        (r.alpha_dihedral_score + r.beta_dihedral_score) / max_val
    )

    return r
