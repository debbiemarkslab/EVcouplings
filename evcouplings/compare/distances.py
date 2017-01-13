"""
Distance calculations on PDB 3D coordinates

Authors:
  Thomas A. Hopf
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def _distances(residues_i, coords_i, residues_j, coords_j, symmetric):
    """
    Compute minimum atom distances between residues. If used on
    a single atom per residue, this function can e.g. also compute
    C_alpha distances.

    Parameters
    ----------
    residues_i : np.array
        Matrix of size N_i x 2, where N_i = number of residues
        in PDB chain used for first axis. Each row of this
        matrix contains the first and last (inclusive) index
        of the atoms comprising this residue in the coords_i
        matrix
    coords_i : np.array
        N_a x 3 matrix containing 3D coordinates of all atoms
        (where N_a is total number of atoms in chain)
    residues_j : np.array
        Like residues_i, but for chain used on second axis
    coords_j : np.array
        Like coords_j, but for chain used on second axis

    Returns
    -------
    dists : np.array
        Matrix of size N_i x N_j containing minimum atom
        distance between residue i and j in dists[i, j]
    """
    LARGE_DIST = 1000000

    N_i, _ = residues_i.shape
    N_j, _ = residues_j.shape

    # matrix to hold final distances
    dists = np.zeros((N_i, N_j))

    # iterate all pairs of residues
    for i in range(N_i):
        for j in range(N_j):
            # limit computation in symmetric case and
            # use previously calculated distance
            if symmetric and i >= j:
                dists[i, j] = dists[j, i]
            else:
                range_i = residues_i[i]
                range_j = residues_j[j]
                min_dist = LARGE_DIST

                # iterate all pairs of atoms for residue pair;
                # end of coord range is inclusive, so have to add 1
                for a_i in range(range_i[0], range_i[1] + 1):
                    for a_j in range(range_j[0], range_j[1] + 1):
                        # compute Euclidean distance between atom pair
                        cur_dist = np.sqrt(
                            np.sum(
                                (coords_i[a_i] - coords_j[a_j]) ** 2
                            )
                        )
                        # store if this is a smaller distance
                        min_dist = min(min_dist, cur_dist)

                dists[i, j] = min_dist

    return dists
