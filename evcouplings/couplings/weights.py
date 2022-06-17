import numpy as np
from numba import jit, prange


def save_raw_weights(weights_filtered, filename, skipped_sequences):
    """
    Save one weight per sequence in alignment, even if they were filtered (in which case, save weight = 0)

    Parameters
    ----------
    weights_filtered : numpy.ndarray
        From alignment.set_weights
    filename : str
        Path to output file
    skipped_sequences : array-like
    """
    pass


@jit(nopython=True)
def num_cluster_members_legacy(matrix, identity_threshold):
    """
    Calculate number of sequences in alignment
    within given identity_threshold of each other

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.

    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))

    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_id = 0
            for k in range(L):
                if matrix[i, k] == matrix[j, k]:
                    pair_id += 1

            if pair_id / L >= identity_threshold:
                num_neighbors[i] += 1
                num_neighbors[j] += 1

    return num_neighbors


@jit(nopython=True, parallel=True)
def num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """
    Calculate number of sequences in alignment
    within given identity_threshold of each other

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
        Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length

    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1  # num_neighbors_i = 0  # TODO why did I make this 0 again? Probably because I thought I'd have to count i == j
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:  # Edit(Lood): Don't count gaps as matches
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            if pair_matches / L_non_gaps[i] >= identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors


@jit(nopython=True)
def num_cluster_members_nogaps_serial(matrix, identity_threshold, invalid_value):
    """
    Calculate number of sequences in alignment
    within given identity_threshold of each other

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
            Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length

    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in range(N):
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:  # Edit(Lood): Don't count gaps as matches
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            if pair_matches / L_non_gaps[i] >= identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors
