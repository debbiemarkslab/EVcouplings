import numpy as np
import numba
import copy


def frequencies(alignment):
    """
    Compute relative single-site frequencies of
    symbols in the alignment.

    Be sure that alignment.set_weights()
    has been called prior to the use of
    this method.

    Implementation note: Frequencies can also be
    computed using the alignment method / property
    frequencies. However, DCA requires to normalize
    the frequencies by the number of effective
    sequences rather than N.

    Parameters
    ----------
    alignment : Alignment
        The alignment. Be sure that alignment.set_weights()
        has been called.

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all symbols.
    """
    matrix = alignment.matrix_mapped
    N, L = matrix.shape

    fi = np.zeros((L, alignment.num_symbols))
    for s in range(N):
        for i in range(L):
            fi[i, matrix[s, i]] += alignment.weights[s]

    # normalize frequencies by the number
    # of effective sequences
    return fi / alignment.weights.sum()


def pair_frequencies(alignment, fi):
    """
    Compute relative pairwise frequencies of
    symbols in the alignment.

    Be sure that alignment.set_weights()
    has been called prior to the use of
    this method.

    Parameters
    ----------
    alignment : Alignment
        The alignment. Be sure that alignment.set_weights()
        has been called.
    fi : np.array
        Matrix of size L x num_symbols containing relative
        single-site frequencies of all characters. The
        pairwise frequencies concerning positions (i, i)
        and symbols (alpha, alpha) will be set to the
        respective single-site frequency of alpha in
        position i.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative frequencies of all possible
        combinations of symbol pairs.
    """
    matrix = alignment.matrix_mapped
    N, L = matrix.shape

    num_symbols = alignment.num_symbols
    fij = np.zeros((L, L, num_symbols, num_symbols))
    for s in range(N):
        for i in range(L):
            for j in range(i + 1, L):
                fij[i, j, matrix[s, i], matrix[s, j]] += alignment.weights[s]
                fij[j, i, matrix[s, j], matrix[s, i]] = fij[i, j, matrix[s, i], matrix[s, j]]

    # normalize frequencies by the number
    # of effective sequences
    fij /= alignment.weights.sum()

    # set the frequencies of symbol pairs (alpha, alpha)
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for alpha in range(num_symbols):
            fij[i, i, alpha, alpha] = fi[i, alpha]

    return fij


def add_pseudo_count_to_frequencies(fi, pseudo_count=0.5):
    """
    Add pseudo-count to single-site frequencies to regularize
    in the case of insufficient data availability.

    Parameters
    ----------
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all symbols.
    pseudo_count : float, optional (default: 0.5)
        The value to be added as pseudo-count.

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing
        single-site frequencies corrected by the
        pseudo-count.
    """
    num_symbols = fi.shape[1]
    return (
        (1. - pseudo_count) * fi +
        (pseudo_count / float(num_symbols))
    )


def add_pseudo_count_to_pair_frequencies(fij, pseudo_count=0.5):
    """
    Add pseudo-count to pairwise frequencies to regularize
    in the case of insufficient data availability.

    Parameters
    ----------
    fij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative pairwise frequencies of all symbols.
    pseudo_count : float, optional (default: 0.5)
        The value to be added as pseudo-count.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing pairwise frequencies corrected by the
        pseudo-count.
    """
    fij_copy = copy.deepcopy(fij)

    L, _, num_symbols, _ = fij.shape

    # add a pseudo-count to the frequencies
    fij = (
        (1. - pseudo_count) * fij +
        (pseudo_count / float(num_symbols ** 2))
    )

    # again, set the "pair" frequency of two identical
    # symbols in the same position to the respective
    # single-site frequency (stored in saved copy of raw fij)
    # also, set all other entries in matrix of position pair
    # (i, i) to zero
    id_matrix = np.identity(num_symbols)
    for i in range(L):
        for alpha in range(num_symbols):
            for beta in range(num_symbols):
                fij[i, i, alpha, beta] = (
                    (1. - pseudo_count) * fij_copy[i, i, alpha, beta] +
                    (pseudo_count / num_symbols) * id_matrix[alpha, beta]
                )

    return fij


@numba.jit(nopython=True)
def _flatten_index(i, alpha, num_symbols):
    """
    Map position and symbol to index in
    the covariance matrix.

    Parameters
    ----------
    i : int, np.array of int
        The alignment column(s).
    alpha : int, np.array of int
        The symbol(s).
    num_symbols : int
        The total number of symbols.
    """
    return i * (num_symbols - 1) + alpha


def compute_covariance_matrix(fi, fij):
    """
    Compute the covariance matrix.

    Parameters
    ----------
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all symbols.
    fij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative frequencies of all possible
        combinations of symbol pairs.

    Returns
    -------
    np.array
        Matrix of size (L * (num_symbols-1)) x (L * (num_symbols-1))
        containing the co-variation of each character pair
        in any positions.
    """
    L, num_symbols = fi.shape

    # The covariance values concerning the last symbol
    # are required to equal zero and are not represented
    # in the covariance matrix - resulting in a matrix of
    # size (L * (num_symbols-1)) x (L * (num_symbols-1))
    # rather than (L * num_symbols) x (L * num_symbols).
    covariance_matrix = np.zeros(
        (L * (num_symbols - 1), L * (num_symbols - 1))
    )

    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    covariance_matrix[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols),
                    ] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j, beta]

    return covariance_matrix


def tilde_fields(i, j, e_ij, fi):
    """Compute h_tilde fields of the two-site model.

    Parameters
    ----------
    i : int
        Position.
    j : int
        Position.
    e_ij : np.array
        Matrix of size num_symbols x num_symbols
        containing all coupling strengths of
        position pair (i, j).
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all symbols.

    Returns
    -------
    np.array, np.array
        h_tilde fields of position i and j -
        both arrays of size 1 x num_symbols
    """
    _EPSILON = 1e-4
    diff = 1.0

    num_symbols = fi.shape[1]
    h_tilde_i = np.full((1, num_symbols), 1 / float(num_symbols))
    h_tilde_j = np.full((1, num_symbols), 1 / float(num_symbols))

    while diff > _EPSILON:
        tmp_1 = np.dot(h_tilde_j, e_ij.T)
        tmp_2 = np.dot(h_tilde_i, e_ij)

        h_tilde_i_updated = fi[i] / tmp_1
        h_tilde_i_updated /= h_tilde_i_updated.sum()

        h_tilde_j_updated = fi[j] / tmp_2
        h_tilde_j_updated /= h_tilde_j_updated.sum()

        diff = max(
            np.absolute(h_tilde_i_updated - h_tilde_i).max(),
            np.absolute(h_tilde_j_updated - h_tilde_j).max()
        )

        h_tilde_i = h_tilde_i_updated
        h_tilde_j = h_tilde_j_updated

    return h_tilde_i, h_tilde_j


def direct_information(inv_c, fi):
    """
    Compute direct information of all possible
    position pairs.

    Parameters
    ----------
    inv_c : np.array
        The inverse of the covariance matrix.
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.

    Returns
    -------
    np.array
        Matrix of size L x L containing the direct
        information of all possible position pairs.
    """
    L, num_symbols = fi.shape

    di = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            # extract entries relevant to position pair (i, j)
            # from the inverted covariance matrix
            # and  apply the exponential to their negatives
            # (e_ij of the last symbol is set to 1,
            # since ln(0) = 1)
            e_ij = np.ones((num_symbols, num_symbols))
            e_ij[: num_symbols - 1, : num_symbols - 1] = np.exp(-inv_c[np.ix_(
                _flatten_index(i, np.arange(num_symbols - 1), num_symbols),
                _flatten_index(j, np.arange(num_symbols - 1), num_symbols)
            )])

            # compute two-site model
            h_tilde_i, h_tilde_j = tilde_fields(i, j, e_ij, fi)
            p_di_ij = e_ij * np.dot(h_tilde_i.T, h_tilde_j)
            p_di_ij = p_di_ij / p_di_ij.sum()

            # dot product of single-site frequencies
            # of columns i and j
            f_ij = np.dot(
                fi[i].reshape((1, 21)).T,
                fi[j].reshape((1, 21))
            )

            # finally, compute direct information as
            # mutual information associated to p_di_ij
            _TINY = 1.0e-100
            di[i, j] = di[j, i] = np.trace(
                np.dot(
                    p_di_ij.T,
                    np.log((p_di_ij + _TINY) / (f_ij + _TINY)))
            )

    return di


def write_raw_ec_file(di, couplings_file, target_sequence, segment):
    """
    Write direct information to couplings file.

    Parameters
    ----------
    di : np.array
        Matrix of size L x L containing the direct
        information of all possible position pairs.
    couplings_file : str
        Output path for file with evolutionary couplings.
    target_sequence : np.array
        Array of length L containing the target sequence.
    segment : Segment
        For now, segment is needed to map index to
        UniProt space.
    """
    L = di.shape[0]
    with open(couplings_file, "w") as f:
        for i in range(L):
            for j in range(i + 1, L):
                f.write(" ".join(map(str, [
                    segment.positions[i],
                    target_sequence[i],
                    segment.positions[j],
                    target_sequence[j],
                    0,
                    "{0:.6f}".format(di[i, j])
                ])) + "\n")


def reshape_inv_covariance_matrix(inv_c, L, num_symbols):
    """
    "Un-flatten" inverse of the covariance matrix to
    allow random and easy access using position and
    symbol indices.

    Parameters
    ----------
    inv_c : np.array
        The inverse of the covariance matrix.
    L : int
        The length of the target sequence,
        i.e. the number of alignment positions.
    num_symbols : int
        The total number of symbols.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing the exact same entries as the input
        matrix.
    """
    eij = np.zeros((L, L, num_symbols, num_symbols))
    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    eij[i, j, alpha, beta] = inv_c[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols)
                    ]
    return eij


def fields(inv_c, fi):
    """
    Compute fields.

    Parameters
    ----------
    inv_c : np.array
        The inverse of the covariance matrix.
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing
        single-site biases.
    """
    L, num_symbols = fi.shape

    hi = np.zeros((L, num_symbols))
    for i in range(L):
        log_fi = np.log(fi[i] / fi[i, num_symbols - 1])
        e_ij_sum = np.zeros((1, num_symbols))
        for j in range(L):
            if i != j:
                # extract entries relevant to position pair (i, j)
                # from the inverted covariance matrix
                e_ij = np.zeros((num_symbols, num_symbols))
                e_ij[: num_symbols - 1, : num_symbols - 1] = -inv_c[np.ix_(
                    _flatten_index(i, np.arange(num_symbols - 1), num_symbols),
                    _flatten_index(j, np.arange(num_symbols - 1), num_symbols)
                )]

                # for position i, some eij values over all j
                e_ij_sum += np.dot(e_ij, fi[j].reshape((1, 21)).T).T
        hi[i] = log_fi - e_ij_sum

    return hi


# TODO: THIS METHOD DOES NOT WORK YET
# problem: __read_plmc_v2 fails to read in
# the dumped target sequence
def write_param_file(param_file, eij, hi, fij, fi,
                     alignment, segment, theta, N_invalid,
                     precision="float32"):
    """
    Write binary Jij file.

    The file format corresponds to the Jij format
    of plmc. Parameters needed for plmc but not
    for DCA (e.g. lambda_h, lambda_J,...) are
    assigned the value -1. DCA's parameter
    pseudo_count is not represented in the Jij file.

    Parameters
    ----------
    param_file : str
        Output path for binary file containing model.
    eij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing coupling strengths for every possible
        combination of symbol and position pairs.
    hi : np.array
        Matrix of size L x num_symbols containing
        the fields.
    fij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative frequencies of all possible
        combinations of symbol pairs.
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all symbols.
    alignment : Alignment
        The input alignment.
    segment : Segment
        The alignment's segment.
    theta : float
        Pairwise identity threshold that was used to cluster and
        weight sequences.
    precision : {"float32", "float64"}
        Write binary file in single or double precision.
    """
    with open(param_file, "w") as f:
        # model length, number of symbols, valid and invalid sequences, iterations
        np.array([alignment.L,
                  alignment.num_symbols,
                  alignment.N,
                  N_invalid,
                  -1], dtype="int32").tofile(f)

        # theta, regularization weights (plmc specific parameters),
        # effective number of sequences
        np.array([theta,
                  -1,  # lambda_h placeholder
                  -1,  # lambda_J placeholder
                  -1,  # lambda_group placeholder
                  alignment.weights.sum()], dtype=precision).tofile(f)

        # alphabet
        alphabet = np.array(list(alignment.alphabet), dtype="S1")
        alphabet[np.where(alphabet != b"")].tofile(f)

        # weights of individual sequences (after clustering)
        alignment.weights.astype(precision).tofile(f)

        # target sequence
        target_seq = alignment.matrix[0].astype("S1")
        target_seq[np.where(target_seq != b"")].tofile(f)

        # index mapping
        np.array(segment.positions, dtype="int32").tofile(f)

        # single site frequencies fi and fields hi
        fi.astype(precision).tofile(f)
        hi.astype(precision).tofile(f)

        # pair frequencies fij
        for i in range(alignment.L - 1):
            for j in range(i + 1, alignment.L):
                fij[i, j].astype(precision).tofile(f)

        # pair couplings eij
        for i in range(alignment.L - 1):
            for j in range(i + 1, alignment.L):
                eij[i, j].astype(precision).tofile(f)
