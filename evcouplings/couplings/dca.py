"""
Inference of evolutionary couplings
from sequence alignments using
mean field approximation.

Authors:
  Sophia F. Mersmann
"""

import numpy as np


def _prepare_alignment(alignment):
    """
    Prepare the input a2m alignment for
    mean field direct coupling analysis.

    The method extracts an alignment of only
    valid sequences and focus alignment columns.

    Parameters
    ----------
    alignment : Alignment
        Alignment with upper and lower letter
        (e.g. a2m alignment). The target sequence
        must be the first record in the alignment.

    Returns
    -------
    Alignment
        Focus alignment without invalid sequences
        or lower letter columns.
    """
    # the first sequence of an a2m alignment
    # in focus mode is the target sequence
    target_seq = alignment[0]

    # select focus columns as alignment columns
    # that are non-gapped and a upper
    # character in the target sequence
    focus_cols = np.array([
        c.isupper() and c not in [
            alignment._match_gap,
            alignment._insert_gap
        ]
        for c in target_seq
    ])

    # extract focus alignment
    focus_ali = alignment.select(columns=focus_cols)

    # find sequences that are valid,
    # i.e. contain only alphabet symbols
    np_alphabet = np.array(list(focus_ali.alphabet))
    valid_sequences = np.array([
        np.in1d(seq, np_alphabet).all()
        for seq in focus_ali.matrix
    ])

    # extract alignment with valid sequences only
    focus_ali = focus_ali.select(sequences=valid_sequences)

    # return focus alignment
    return focus_ali


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
        The number of symbols of the
        alphabet used.
    """
    return i * (num_symbols - 1) + alpha


def _tilde_fields(e_ij, f_i, f_j):
    """Compute h_tilde fields of the two-site model.

    Parameters
    ----------
    e_ij : np.array
        Matrix of size num_symbols x num_symbols
        containing all coupling strengths of
        position pair (i, j).
    f_i : np.array
        Row i of single-site frequencies.
    f_j : np.array
        Row j of single-site frequencies.

    Returns
    -------
    np.array, np.array
        h_tilde fields of position i and j -
        both arrays of size 1 x num_symbols
    """
    length = f_i.shape[0]

    _EPSILON = 1e-4
    diff = 1.0

    h_tilde_i = np.full((1, length), 1 / float(length))
    h_tilde_j = np.full((1, length), 1 / float(length))

    while diff > _EPSILON:
        tmp_1 = np.dot(h_tilde_j, e_ij.T)
        tmp_2 = np.dot(h_tilde_i, e_ij)

        h_tilde_i_updated = f_i / tmp_1
        h_tilde_i_updated /= h_tilde_i_updated.sum()

        h_tilde_j_updated = f_j / tmp_2
        h_tilde_j_updated /= h_tilde_j_updated.sum()

        diff = max(
            np.absolute(h_tilde_i_updated - h_tilde_i).max(),
            np.absolute(h_tilde_j_updated - h_tilde_j).max()
        )

        h_tilde_i = h_tilde_i_updated
        h_tilde_j = h_tilde_j_updated

    return h_tilde_i, h_tilde_j


def reshape_invC_to_4d(inv_c, L, num_symbols):
    """
    "Un-flatten" inverse of the covariance
    matrix to allow easy access to couplings
    using position and symbol indices.

    Parameters
    ----------
    inv_c : np.array
        The inverse of the covariance matrix
        of shape (L * (num_symbols - 1))**2.
    L : int
        The width of the alignment.
    num_symbols : int
        The number of symbols of the alphabet.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols.
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


class MeanFieldDirectCouplingAnalysis:
    """
    Class that provides the functionality
    to infer evolutionary couplings from a given
    sequence alignment using mean field
    approximation.

    Important:
    The input alignment should be an a2m
    alignment with lower / upper columns
    and the target sequence as the first
    record.

    Attributes
    ----------
    _raw_alignment : Alignment
        The input alignment. This should be an
        a2m alignment with lower / upper columns
        and the target sequence as first record.
    alignment : Alignment
        A processed version of the given alignment
        (_raw_alignment) that is then used to
        infer evolutionary couplings using DCA.
    N : int
        The number of sequences (of the processed
        alignment).
    L : int
        The width of the alignment (again, this
        refers to the processed alignment).
    num_symbols : int
        The number of symbols of the alphabet used.
    raw_fi : np.array
        Matrix of size L x num_symbols containing
        relative column frequencies of all symbols.
    fi : np.array
        raw_fi corrected by a pseudo-count.
    raw_fij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative frequencies of all possible
        combinations of symbol pairs.
    fij : np.array
        raw_fij corrected by a pseudo-count.
    covariance_matrix : np.array
        Matrix of size (L * (num_symbols-1)) x (L * (num_symbols-1))
        containing the co-variation of each character pair
        in any positions.
    hi : np.array
        Matrix of size L x num_symbols containing
        single-site fields.
    eij : np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing coupling strengths.
    mi : np.array
        Matrix of size L x L containing the mutual
        information of all possible position pairs.
    di : np.array
        Matrix of size L x L containing the direct
        information of all possible position pairs.
    """
    def __init__(self, alignment):
        """
        Initialize direct couplings analysis by
        processing the given alignment.

        Parameters
        ----------
        alignment : Alignment
            Alignment with lower / upper columns
            and the target sequence as first record.
        """
        # input alignment
        self._raw_alignment = alignment

        # prepare alignment for DCA
        # by removing invalid sequences
        # and selecting focus columns
        self.alignment = _prepare_alignment(self._raw_alignment)

        # for convenience
        self.N = self.alignment.N
        self.L = self.alignment.L
        self.num_symbols = self.alignment.num_symbols

        # raw and corrected single-site frequencies
        self.raw_fi = None
        self.fi = None

        # raw and corrected pair frequencies
        self.raw_fij = None
        self.fij = None

        # covariance matrix
        self.covariance_matrix = None

        # fields and couplings
        self.hi = None
        self.eij = None

        # mutual and direct information
        self.mi = None
        self.di = None

    @classmethod
    def run(cls, alignment, theta=0.8, pseudo_count=0.5):
        """
        Run mean field direct couplings analysis
        on a given alignment.

        Parameters
        ----------
        alignment : Alignment
            Alignment with lower / upper columns
            and the target sequence as first record.
        theta : float, optional (default: 0.8)
            Sequences with pairwise identity >= theta
            will be clustered and their sequence weights
            downweighted as 1 / num_cluster_members.
        pseudo_count : float, optional (default: 0.5)
            Applied to frequency counts to regularize
            in the case of insufficient data availability.

        Returns
        -------
        MeanFieldDirectCouplingAnalysis
            DCA object that holds computed mutual and
            direct information as well as evolutionary
            coupling strengths, single-site fields,
            frequencies and the underlying covariance matrix.
        """
        # init mean field DCA
        dca = cls(alignment)

        # make sure sequence weights exist
        if dca.alignment.weights is None:
            dca.alignment.set_weights(identity_threshold=theta)

        # compute relative single-site frequencies
        # of symbols in the alignment
        dca.frequencies()

        # compute relative pairwise frequencies
        # of symbols in the alignment
        dca.pair_frequencies()

        # add pseudo-count to column frequencies
        dca.add_pseudo_count_to_frequencies(
            pseudo_count=pseudo_count
        )

        # add pseudo-count to pairwise frequencies
        dca.add_pseudo_count_to_pair_frequencies(
            pseudo_count=pseudo_count
        )

        # compute the covariance matrix from
        # the column and pair frequencies
        dca.compute_covariance_matrix()

        # coupling parameters are inferred
        # by inverting the covariance matrix
        inv_cov_matrix = np.linalg.inv(dca.covariance_matrix)

        # reshape the inverse of the covariance matrix
        # to make eijs easily accessible
        dca.eij = reshape_invC_to_4d(
            inv_cov_matrix, dca.L, dca.num_symbols
        )

        # compute fields
        dca.fields()

        # compute mutual information from raw
        # relative frequencies
        dca.mutual_information()

        # compute direct information
        dca.direct_information()

        return dca

    def frequencies(self):
        """
        Compute relative single-site frequencies of
        symbols in the alignment.

        This method sets the attribute self.raw_fi
        and returns a reference to it.

        Be sure that alignment.set_weights()
        has been called prior to the use of
        this method.

        Implementation note: Frequencies can also be
        computed using the alignment method / property
        frequencies. However, DCA requires to normalize
        the frequencies by the number of effective
        sequences rather than N.

        Returns
        -------
        np.array
            Reference to attribute self.raw_fi
        """
        matrix = self.alignment.matrix_mapped

        self.raw_fi = np.zeros((self.L, self.num_symbols))
        for s in range(self.N):
            for i in range(self.L):
                self.raw_fi[i, matrix[s, i]] += self.alignment.weights[s]

        # normalize frequencies by the number
        # of effective sequences
        self.raw_fi /= self.alignment.weights.sum()

        return self.raw_fi

    def pair_frequencies(self):
        """
        Compute relative pairwise frequencies of
        symbols in the alignment.

        This method sets the attribute self.raw_fij
        and returns a reference to it.

        Be sure that alignment.set_weights()
        has been called prior to the use of
        this method.

        Returns
        -------
        np.array
            Reference to attribute self.raw_fij
        """
        matrix = self.alignment.matrix_mapped

        self.raw_fij = np.zeros((self.L, self.L, self.num_symbols, self.num_symbols))
        for s in range(self.N):
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    self.raw_fij[i, j, matrix[s, i], matrix[s, j]] += self.alignment.weights[s]
                    self.raw_fij[j, i, matrix[s, j], matrix[s, i]] = self.raw_fij[i, j, matrix[s, i], matrix[s, j]]

        # normalize frequencies by the number
        # of effective sequences
        self.raw_fij /= self.alignment.weights.sum()

        # set the frequencies of symbol pairs (alpha, alpha)
        # in position i to the respective single-site
        # frequency of alpha in position i
        for i in range(self.L):
            for alpha in range(self.num_symbols):
                self.raw_fij[i, i, alpha, alpha] = self.raw_fi[i, alpha]

        return self.raw_fij

    def mutual_information(self):
        """
        Compute mutual information
        of all possible position pairs.

        This method sets the attribute self.mi
        and returns a reference to it.

        Returns
        -------
        np.array
            Reference to attribute self.mi
        """
        self.mi = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                for alpha in range(self.num_symbols):
                    for beta in range(self.num_symbols):
                        if self.raw_fij[i, j, alpha, beta] > 0:
                            self.mi[i, j] += (
                                self.raw_fij[i, j, alpha, beta] *
                                np.log(
                                    self.raw_fij[i, j, alpha, beta] / (self.raw_fi[i, alpha] * self.raw_fi[j, beta])
                                )
                            )

        return self.mi

    def add_pseudo_count_to_frequencies(self, pseudo_count=0.5):
        """
        Add pseudo-count to the raw single-site frequencies
        to regularize in the case of insufficient data availability.

        This method sets the attribute self.fi
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Reference to attribute self.fi
        """
        self.fi = (
            (1. - pseudo_count) * self.raw_fi +
            (pseudo_count / float(self.num_symbols))
        )

        return self.fi

    def add_pseudo_count_to_pair_frequencies(self, pseudo_count=0.5):
        """
        Add pseudo-count to pairwise frequencies to regularize
        in the case of insufficient data availability.

        This method sets the attribute self.fij
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Reference to attribute self.fij
        """
        # add a pseudo-count to the frequencies
        self.fij = (
            (1. - pseudo_count) * self.raw_fij +
            (pseudo_count / float(self.num_symbols ** 2))
        )

        # again, set the "pair" frequency of two identical
        # symbols in the same position to the respective
        # single-site frequency (stored in saved copy of raw fij)
        # also, set all other entries in matrix of position pair
        # (i, i) to zero
        id_matrix = np.identity(self.num_symbols)
        for i in range(self.L):
            for alpha in range(self.num_symbols):
                for beta in range(self.num_symbols):
                    self.fij[i, i, alpha, beta] = (
                        (1. - pseudo_count) * self.raw_fij[i, i, alpha, beta] +
                        (pseudo_count / self.num_symbols) * id_matrix[alpha, beta]
                    )

        return self.fij

    def compute_covariance_matrix(self):
        """
        Compute the covariance matrix.

        This method sets the attribute self.covariance_matrix
        and returns a reference to it.

        Returns
        -------
        np.array
            Reference to attribute self.convariance_matrix
        """
        # The covariance values concerning the last symbol
        # are required to equal zero and are not represented
        # in the covariance matrix (important for taking the
        # inverse) - resulting in a matrix of size
        # (L * (num_symbols-1)) x (L * (num_symbols-1))
        # rather than (L * num_symbols) x (L * num_symbols).
        self.covariance_matrix = np.zeros(
            (self.L * (self.num_symbols - 1),
             self.L * (self.num_symbols - 1))
        )

        for i in range(self.L):
            for j in range(self.L):
                for alpha in range(self.num_symbols - 1):
                    for beta in range(self.num_symbols - 1):
                        self.covariance_matrix[
                            _flatten_index(i, alpha, self.num_symbols),
                            _flatten_index(j, beta, self.num_symbols),
                        ] = self.fij[i, j, alpha, beta] - self.fi[i, alpha] * self.fi[j, beta]

        return self.covariance_matrix

    def direct_information(self):
        """
        Compute direct information of all possible
        position pairs.

        This method sets the attribute self.di
        and returns a reference to it.

        Returns
        -------
        np.array
            Reference to attribute self.di
        """
        self.di = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                # extract couplings relevant to
                # position pair (i, j)
                e_ij = np.exp(-self.eij[i, j])

                # compute two-site model
                h_tilde_i, h_tilde_j = _tilde_fields(e_ij, self.fi[i], self.fi[j])
                p_di_ij = e_ij * np.dot(h_tilde_i.T, h_tilde_j)
                p_di_ij = p_di_ij / p_di_ij.sum()

                # dot product of single-site frequencies
                # of columns i and j
                f_ij = np.dot(
                    self.fi[i].reshape((1, 21)).T,
                    self.fi[j].reshape((1, 21))
                )

                # finally, compute direct information as
                # mutual information associated to p_di_ij
                _TINY = 1.0e-100
                self.di[i, j] = self.di[j, i] = np.trace(
                    np.dot(
                        p_di_ij.T,
                        np.log((p_di_ij + _TINY) / (f_ij + _TINY)))
                )

        return self.di

    def to_raw_ec_file(self, couplings_file, segment):
        """
        Write mutual and direct information to the EC file.

        Parameters
        ----------
        couplings_file : str
            Output path for file with evolutionary couplings.
        segment : Segment
            For now, segment is needed to map index to
            UniProt space.
        """
        target_sequence = self.alignment.matrix[0]

        with open(couplings_file, "w") as f:
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    f.write(" ".join(map(str, [
                        segment.positions[i],
                        target_sequence[i],
                        segment.positions[j],
                        target_sequence[j],
                        "{0:.6f}".format(self.mi[i, j]),
                        "{0:.6f}".format(self.di[i, j])
                    ])) + "\n")

    def fields(self):
        """
        Compute fields.

        This method sets the attribute self.hi
        and returns a reference to it.

        Returns
        -------
        np.array
            Reference to attribute self.hi
        """
        self.hi = np.zeros((self.L, self.num_symbols))
        for i in range(self.L):
            log_fi = np.log(self.fi[i] / self.fi[i, self.num_symbols - 1])
            e_ij_sum = np.zeros((1, self.num_symbols))
            for j in range(self.L):
                if i != j:
                    # extract couplings relevant to position pair (i, j)
                    e_ij = -self.eij[i, j]

                    # some eij values over all j
                    e_ij_sum += np.dot(e_ij, self.fi[j].reshape((1, 21)).T).T
            self.hi[i] = log_fi - e_ij_sum

        return self.hi

    def to_model_file(self, param_file, segment, theta,
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
        segment : Segment
            The alignment's segment.
        theta : float
            Pairwise identity threshold that was used to cluster and
            weight sequences.
        precision : {"float32", "float64"}
            Write binary file in single or double precision.
        """
        with open(param_file, "w") as f:
            # model length, number of symbols, valid and invalid sequences
            # and a placeholder for iterations (plmc specific parameter)
            np.array([self.L,
                      self.num_symbols,
                      self.N,
                      # for now, the number of invalid sequences
                      # is set to zero since __read_plmc_v2() in
                      # couplings.model reads in the weights of
                      # valid and invalid sequences but here only
                      # weights of the valid sequences are used -
                      # so, the number of invalid sequences must be
                      # zero to prevent reading the model file
                      # from crashing
                      # self._raw_alignment.N - self.N,
                      0,
                      -1], dtype="int32").tofile(f)

            # theta, regularization weights (plmc specific parameters)
            # and the effective number of sequences
            np.array([theta,
                      -1,  # lambda_h placeholder
                      -1,  # lambda_J placeholder
                      -1,  # lambda_group placeholder
                      self.alignment.weights.sum()], dtype=precision).tofile(f)

            # alphabet
            alphabet = np.array(list(self.alignment.alphabet), dtype="S1")
            alphabet[np.where(alphabet != b"")].tofile(f)

            # weights of individual sequences (after clustering)
            self.alignment.weights.astype(precision).tofile(f)

            # target sequence
            target_seq = self.alignment.matrix[0].astype("S1")
            target_seq[np.where(target_seq != b"")].tofile(f)

            # index mapping
            np.array(segment.positions, dtype="int32").tofile(f)

            # single site frequencies fi and fields hi
            self.fi.astype(precision).tofile(f)
            self.hi.astype(precision).tofile(f)

            # pair frequencies fij
            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.fij[i, j].astype(precision).tofile(f)

            # pair couplings eij
            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.eij[i, j].astype(precision).tofile(f)
