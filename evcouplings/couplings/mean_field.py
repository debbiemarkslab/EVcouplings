"""
Inference of evolutionary couplings
from sequence alignments using
mean field approximation.

Authors:
  Sophia F. Mersmann
"""

import numpy as np
import numba
from copy import deepcopy

from evcouplings.align import parse_header
from evcouplings.couplings import CouplingsModel

# arbitrary value that is written
# to file for plmc-specific parameters
_PLACEHOLDER = -1


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
        The number of symbols of the
        alphabet used.
    """
    return i * (num_symbols - 1) + alpha


class MeanFieldDCA:
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
    index_list : np.array
        List of UniProt numbers of the target
        sequence (only upper case characters).
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
    covariance_matrix : np.array
        Matrix of size (L * (num_symbols-1)) x (L * (num_symbols-1))
        containing the co-variation of each character pair
        in any positions.
    covariance_matrix_inv : np.array
        Inverse of covariance_matrix.
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

        # the first sequence of an a2m alignment
        # in focus mode is the target sequence
        target_seq = self._raw_alignment[0]

        # select focus columns as alignment columns
        # that are non-gapped and a upper
        # character in the target sequence
        focus_cols = np.array([
            c.isupper() and c not in [
                self._raw_alignment._match_gap,
                self._raw_alignment._insert_gap
            ]
            for c in target_seq
        ])

        # extract focus alignment
        focus_ali = self._raw_alignment.select(
            columns=focus_cols
        )

        # extract index list of the target sequence
        # (only focus columns)
        _, start, stop = parse_header(self._raw_alignment.ids[0])
        self.index_list = np.array(range(start, stop + 1))
        self.index_list = self.index_list[focus_cols]

        # find sequences that are valid,
        # i.e. contain only alphabet symbols
        np_alphabet = np.array(list(focus_ali.alphabet))
        valid_sequences = np.array([
            np.in1d(seq, np_alphabet).all()
            for seq in focus_ali.matrix
        ])

        # remove invalid sequences
        self.alignment = focus_ali.select(
            sequences=valid_sequences
        )

        # reset pre-calculated sequence weigths
        # and frequencies of the alignment
        self._reset()

    def _reset(self):
        """
        Reset pre-computed sequence weights and
        alignment frequencies as well as the
        covariance matrix and its inverse.

        Resetting becomes important, when the
        fit function is called more than once.
        """
        # reset theta-specific weights
        self.alignment.weights = None

        # also reset frequencies since these
        # were based on the weights (and the
        # given pseudo-count)
        self.alignment._frequencies = None
        self.alignment._pair_frequencies = None
        self.regularized_frequencies = None
        self.regularized_pair_frequencies = None

        # reset covariance matrix and its inverse
        self.covariance_matrix = None
        self.covariance_matrix_inv = None

    def fit(self, theta=0.8, pseudo_count=0.5):
        """
        Run mean field direct couplings analysis.

        Parameters
        ----------
        theta : float, optional (default: 0.8)
            Sequences with pairwise identity >= theta
            will be clustered and their sequence weights
            downweighted as 1 / num_cluster_members.
        pseudo_count : float, optional (default: 0.5)
            Applied to frequency counts to regularize
            in the case of insufficient data availability.

        Returns
        -------
        MeanFieldCouplingsModel
            Model object that holds the inferred
            fields (h_i) and couplings (J_ij).
        """
        self._reset()

        # compute sequence weights
        # using the given theta
        self.alignment.set_weights(identity_threshold=theta)

        # compute column frequencies regularized by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.regularize_frequencies(pseudo_count=pseudo_count)

        # compute pairwise frequencies regularized by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.regularize_pair_frequencies(pseudo_count=pseudo_count)

        # compute the covariance matrix from
        # the column and pair frequencies
        self.compute_covariance_matrix()

        # coupling parameters are inferred
        # by inverting the covariance matrix
        self.covariance_matrix_inv = -np.linalg.inv(
            self.covariance_matrix
        )

        # reshape the inverse of the covariance matrix
        # to make eijs easily accessible
        J_ij = self.reshape_invC_to_4d()

        # compute fields
        h_i = self.fields()

        return MeanFieldCouplingsModel(
            alignment=self.alignment,
            index_list=self.index_list,
            regularized_f_i=self.regularized_frequencies,
            regularized_f_ij=self.regularized_pair_frequencies,
            h_i=h_i, J_ij=J_ij,
            theta=theta,
            pseudo_count=pseudo_count
        )

    def regularize_frequencies(self, pseudo_count=0.5):
        """
        Returns single-site frequencies
        regularized by a pseudo-count of symbols
        in alignment.

        This method sets the attribute
        self.regularized_frequencies
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols containing
            relative column frequencies of all symbols
            regularized by a pseudo-count.
        """
        self.regularized_frequencies = regularize_frequencies(
            self.alignment.frequencies,
            pseudo_count=pseudo_count
        )
        return self.regularized_frequencies

    def regularize_pair_frequencies(self, pseudo_count=0.5):
        """
        Add pseudo-count to pairwise frequencies
        to regularize in the case of insufficient
        data availability.

        This method sets the attribute
        self.regularized_pair_frequencies
        and returns a reference to it.

        Parameters
        ----------
        pseudo_count : float, optional (default: 0.5)
            The value to be added as pseudo-count.

        Returns
        -------
        np.array
            Matrix of size L x L x num_symbols x num_symbols
            containing relative pairwise frequencies of all
            symbols regularized by a pseudo-count.
        """
        self.regularized_pair_frequencies = regularize_pair_frequencies(
            self.alignment.pair_frequencies,
            pseudo_count=pseudo_count
        )
        return self.regularized_pair_frequencies

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
        self.covariance_matrix = compute_covariance_matrix(
            self.regularized_frequencies,
            self.regularized_pair_frequencies
        )
        return self.covariance_matrix

    def reshape_invC_to_4d(self):
        """
        "Un-flatten" inverse of the covariance
        matrix to allow easy access to couplings
        using position and symbol indices.

        Returns
        -------
        np.array
            Matrix of size L x L x
            num_symbols x num_symbols.
        """
        return reshape_invC_to_4d(
            self.covariance_matrix_inv,
            self.alignment.L,
            self.alignment.num_symbols
        )

    def fields(self):
        """
        Compute fields.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols
            containing single-site fields.
        """
        return fields(
            self.reshape_invC_to_4d(),
            self.regularized_frequencies
        )


class MeanFieldCouplingsModel(CouplingsModel):
    """
    Mean field DCA specific model class that stores the
    parameters inferred using mean field approximation
    and calculates mutual and direct information as
    well as fn and cn scores.
    """
    def __init__(self, alignment, index_list, regularized_f_i,
                 regularized_f_ij, h_i, J_ij, theta,
                 pseudo_count):
        """
        Initialize the model with the results of a
        mean field inference.

        Parameters
        ----------
        alignment : Alignment
            The alignment that was used inferring
            model parameters using mean field approximation.
        index_list : np.array
            Array of UniProt numbers of the target
            sequence (only upper case characters).
        regularized_f_i : np.array
            Matrix of size L x num_symbols
            containing column frequencies
            corrected by pseudo-count.
        regularized_f_ij : np.array
            Matrix of size L x L x num_symbols x
            num_symbols containing pair
            frequencies corrected by pseudo-count.
        h_i : np.array
            Matrix of size L x num_symbols
            containing single-site fields.
        J_ij : np.array
            Matrix of size L x L x num_symbols x
            num_symbols containing couplings.
        theta : float
            The theta used to compute sequence
            weights in mean field DCA.
        pseudo_count : float
            The pseudo-count that was just to
            adjust single and pairwise frequencies
            in mean field DCA.
        """
        # model length, number of symbols
        # and number of valid sequences
        self.L = alignment.L
        self.num_symbols = alignment.num_symbols
        self.N_valid = alignment.N

        # if sequence weights are not set,
        # assume equal weight for every sequence
        if alignment.weights is None:
            self.weights = np.ones((alignment.N))
        else:
            self.weights = alignment.weights

        # number of effective sequences
        self.N_eff = self.weights.sum()

        # alphabet
        self.alphabet = np.array(list(alignment.alphabet))
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # in a focus alignment, the target sequence
        # is the first record in the alignment
        self.target_seq = list(alignment.matrix[0])

        # raw single and pair frequencies
        self.f_i = alignment.frequencies
        self.f_ij = alignment.pair_frequencies

        # regularized single and pair frequencies
        self.regularized_f_i = regularized_f_i
        self.regularized_f_ij = regularized_f_ij

        # fields and couplings
        self.h_i = h_i
        self.J_ij = J_ij

        # parameters used in mean field inference
        self.theta = theta
        self.pseudo_count = pseudo_count

        # mapping
        self.index_list = index_list

        # for now, the number of invalid sequences
        # is set to zero since __read_plmc_v2() in
        # couplings.model reads in the weights of
        # valid and invalid sequences but here only
        # weights of the valid sequences are used -
        # so, the number of invalid sequences must be
        # zero to prevent reading the model file
        # from crashing
        self.N_invalid = 0

        self.__decode_unused_fields(save_pseudo_count=False)
        self._reset_precomputed()

    def _reset_precomputed(self):
        """
        Delete precomputed values (e.g. mutation matrices)

        This method overrides its respective parent method
        in CouplingsModel. It additionally resets DI scores.
        """
        super(MeanFieldCouplingsModel, self)._reset_precomputed()
        self._di_scores = None

    def _calculate_ecs(self):
        """
        Calculates FN and CN scores as defined in
        Ekeberg et al., Phys Rev E, 2013,
        as well as MI and DI scores.

        This method overrides its respective parent method
        in CouplingsModel.
        First, the parent method is called to calculate
        FN, CN and MI scores. Then, DI scores are
        additionally calculated and added to the resulting
        data frame.

        Returns
        -------
        pd.DataFrame
            ECs data frame with columns i, A_i, j, A_j,
            seqdist, mi_raw, mi_apc, fn, cn, di
        """
        # calculates FN, CN as well as MI scores
        # and stores in ECs data frame
        super(MeanFieldCouplingsModel, self)._calculate_ecs()

        # calculate DI scores
        self._di_scores = direct_information(
            self.J_ij, self.regularized_f_i
        )

        # add DI scores to EC data frame
        di = []
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                di.append(self._di_scores[i, j])

        self._ecs.sort_values(by=["i", "j"], inplace=True)
        self._ecs.loc[:, "di"] = di

        return self._ecs.sort_values(
            by="di", ascending=False
        )

    def regularize_f_i(self):
        """
        Returns single-site frequencies
        regularized by a pseudo-count of
        symbols in alignment.

        This method sets the attribute
        self.regularized_f_i and returns
        a reference to it.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols containing
            relative column frequencies of all symbols
            regularized by a pseudo-count.
        """
        self.regularized_f_i = regularize_frequencies(
            self.f_i, pseudo_count=self.pseudo_count
        )
        return self.regularized_f_i

    def regularize_f_ij(self):
        """
        Returns pairwise frequencies
        regularized by a pseudo-count of
        symbols in alignment.

        This method sets the attribute
        self.regularized_f_ij and returns
        a reference to it.

        Returns
        -------
        np.array
            Matrix of size L x L x num_symbols x num_symbols
            containing relative pairwise frequencies of all
            symbols regularized by a pseudo-count.
        """
        self.regularized_f_ij = regularize_pair_frequencies(
            self.f_ij, pseudo_count=self.pseudo_count
        )
        return self.regularized_f_ij

    def tilde_fields(self, i, j):
        """Compute h_tilde fields of the two-site model.

        Parameters
        ----------
        i : int
            First position.
        j : int
            Second position.

        Returns
        -------
        np.array, np.array
            h_tilde fields of position i and j -
            both arrays of size 1 x num_symbols
        """
        return tilde_fields(
            self.J_ij,
            self.regularized_f_i[i],
            self.regularized_f_ij[j]
        )

    @property
    def di_scores(self):
        """
        L x L numpy matrix with DI (direct information) scores
        """
        if self._di_scores is None:
            self._calculate_ecs()

        return self._di_scores

    def to_independent_model(self):
        """
        Compute a single-site model.

        This method overrides its respective
        parent method in CouplingsModel.

        Returns
        -------
        MeanFieldCouplingsModel
            Copy of object turned into independent model
        """
        c0 = deepcopy(self)
        c0.h_i = np.log(self.regularized_f_i)
        c0.J_ij.fill(0)
        c0._reset_precomputed()
        return c0

    def to_raw_ec_file(self, couplings_file):
        """
        Write mutual and direct information to the EC file.

        Parameters
        ----------
        couplings_file : str
            Output path for file with evolutionary couplings.
        """
        with open(couplings_file, "w") as f:
            for i in range(self.L):
                for j in range(i + 1, self.L):
                    f.write(" ".join(map(str, [
                        self.index_list[i], self.target_seq[i],
                        self.index_list[j], self.target_seq[j],
                        "{0:.6f}".format(self.mi_scores_raw[i, j]),
                        "{0:.6f}".format(self.mi_scores_apc[i, j]),
                        "{0:.6f}".format(self.di_scores[i, j]),
                        "{0:.6f}".format(self.cn_scores[i, j])
                    ])) + "\n")

    def transform_from_plmc_model(self):
        """
        Adaptions that allow to read
        a mean-field couplings model
        from file where __read_plmc_v2
        in CouplingsModel does the
        heavy lifting.

        This includes:
        - Manage unused plmc-specific
        fields as well as the pseudo count
        field
        - Modify pair frequencies
        - Regularize column and pair
        frequencies (i.e. fill the fields
        regularized_f_i and regularized_f_ij)
        """
        self.__decode_unused_fields()

        # set the frequency of a pair (alpha, alpha)
        # in position i to the respective single-site
        # frequency of alpha in position i
        for i in range(self.L):
            for alpha in range(self.num_symbols):
                self.f_ij[i, i, alpha, alpha] = self.f_i[i, alpha]

        # compute pseudo counted frequencies
        # from raw frequencies
        self.regularize_f_i()
        self.regularize_f_ij()

    def __encode_unused_fields(self):
        """
        Set plmc-specific parameters
        (lambda_J, lambda_group, num_iter)
        to an arbitrary numerical value.

        Note:
        The field lambda_h is used to store
        the pseudo count (the negative sign
        simply serves as marker).
        """
        self.lambda_J = _PLACEHOLDER
        self.lambda_group = _PLACEHOLDER
        self.num_iter = _PLACEHOLDER
        self.lambda_h = -self.pseudo_count

    def __decode_unused_fields(self, save_pseudo_count=True):
        """
        Set plmc-specific parameters
        to None to ensure correct usage
        of the object.

        Note:
        If save_pseudo_count is True,
        the pseudo count (that was temporarily
        stored in lambda_h) is stored in the
        appropriate field.

        Parameters
        ----------
        save_pseudo_count : bool, optional (default: True)
            If True, the pseudo count is read from
            the field lambda_h and stored in the
            appropriate field.
        """
        self.lambda_J = None
        self.lambda_group = None
        self.num_iter = None

        if save_pseudo_count:
            self.pseudo_count = -self.lambda_h

        self.lambda_h = None

    def to_file(self, out_file, precision="float32", file_format="plmc_v2"):
        """
        Writes the model to binary file.

        This method overrides its respective
        parent method in CouplingsModel.

        Parameters
        ----------
        out_file: str
            A string specifying the path to a file
        precision: {"float16", "float32", "float64"}, optional (default: "float32")
            Numerical NumPy data type specifying the precision
            used to write numerical values to file
        file_format : {"plmc_v2"}, optional (default: "plmc_v2")
            For now, a mean-field model can only be
            written to a file in plmc_v2 format. Writing
            to a plmc_v1 file is not permitted since
            there is no functionality provided to read a
            mean-field model in plmc_v1 format.
        """
        # plmc-specific parameters need to be set to a
        # numerical value to make the to_file function work
        self.__encode_unused_fields()

        # writing to a file in plmc_v1 format
        # is not permitted
        if file_format == "plmc_v1":
            raise ValueError(
                "Illegal file format: plmc_v1. "
                "Valid option: plmc_v2."
            )

        # write the model to file
        super(MeanFieldCouplingsModel, self).to_file(
            out_file,
            precision=precision,
            file_format=file_format
        )

        # transform model to its proper state again
        self.__decode_unused_fields()


def regularize_frequencies(f_i, pseudo_count=0.5):
    """
    Returns/calculates single-site frequencies
    regularized by a pseudo-count of symbols
    in alignment.

    Parameters
    ----------
    f_i : np.array
        Matrix of size L x num_symbols
        containing column frequencies.
    pseudo_count : float, optional (default: 0.5)
        The value to be added as pseudo-count.

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing
        relative column frequencies of all symbols
        regularized by a pseudo-count.
    """
    _, num_symbols = f_i.shape
    regularized_frequencies = (
        (1. - pseudo_count) * f_i +
        (pseudo_count / float(num_symbols))
    )
    return regularized_frequencies


def regularize_pair_frequencies(f_ij, pseudo_count=0.5):
    """
    Add pseudo-count to pairwise frequencies
    to regularize in the case of insufficient
    data availability.

    Parameters
    ----------
    f_ij : np.array
        Matrix of size L x L x num_symbols x
        num_symbols containing pair frequencies.
    pseudo_count : float, optional (default: 0.5)
        The value to be added as pseudo-count.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols
        containing relative pairwise frequencies of all
        symbols regularized by a pseudo-count.
    """
    L, _, num_symbols, _ = f_ij.shape

    # add a pseudo-count to the frequencies
    regularized_pair_frequencies = (
        (1. - pseudo_count) * f_ij +
        (pseudo_count / float(num_symbols ** 2))
    )

    # again, set the "pair frequency" of two identical
    # symbols in the same position to the respective
    # single-site frequency (and all other entries
    # in matrices concerning position pair (i, i) to zero)
    id_matrix = np.identity(num_symbols)
    for i in range(L):
        for alpha in range(num_symbols):
            for beta in range(num_symbols):
                regularized_pair_frequencies[i, i, alpha, beta] = (
                    (1. - pseudo_count) * f_ij[i, i, alpha, beta] +
                    (pseudo_count / num_symbols) * id_matrix[alpha, beta]
                )

    return regularized_pair_frequencies


@numba.jit(nopython=True)
def tilde_fields(J_ij, f_i, f_j):
    """Compute h_tilde fields of the two-site model.

    Parameters
    ----------
    J_ij : np.array
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
    _EPSILON = 1e-4
    diff = 1.0

    num_symbols = f_i.shape[0]

    h_tilde_i = np.full((1, num_symbols), 1 / float(num_symbols))
    h_tilde_j = np.full((1, num_symbols), 1 / float(num_symbols))

    while diff > _EPSILON:
        tmp_1 = np.dot(h_tilde_j, J_ij.T)
        tmp_2 = np.dot(h_tilde_i, J_ij)

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


@numba.jit(nopython=True)
def direct_information(J_ij, f_i):
    """
    Calculate direct information.

    Parameters
    ----------
    J_ij : np.array
        Matrix of size num_symbols x num_symbols
        containing all coupling strengths of
        position pair (i, j).
    f_i : np.array
        Matrix of size L x num_symbols
        containing column frequencies.

    Returns
    -------
    np.array
        Matrix of size L x L
        containing direct information.
    """
    L, num_symbols = f_i.shape

    di = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            # extract couplings relevant to
            # position pair (i, j)
            J = np.exp(J_ij[i, j])

            # compute two-site model
            h_tilde_i, h_tilde_j = tilde_fields(J, f_i[i], f_i[j])
            p_di_ij = J * np.dot(h_tilde_i.T, h_tilde_j)
            p_di_ij = p_di_ij / p_di_ij.sum()

            # dot product of single-site frequencies
            # of columns i and j
            f_ij = np.dot(
                f_i[i].reshape((1, num_symbols)).T,
                f_i[j].reshape((1, num_symbols))
            )

            # finally, compute direct information as
            # mutual information associated to p_di_ij
            _TINY = 1.0e-100
            di[i, j] = di[j, i] = np.trace(
                np.dot(
                    p_di_ij.T,
                    np.log((p_di_ij + _TINY) / (f_ij + _TINY))
                )
            )

    return di


@numba.jit(nopython=True)
def compute_covariance_matrix(f_i, f_ij):
    """
    Compute the covariance matrix.

    Parameters
    ----------
    f_i : np.array
        Matrix of size L x num_symbols
        containing column frequencies.
    f_ij : np.array
        Matrix of size L x L x num_symbols x
        num_symbols containing pair frequencies.

    Returns
    -------
    np.array
        Matrix of size L x (num_symbols-1) x
        L x (num_symbols-1) containing
        covariance values.
    """
    L, num_symbols = f_i.shape

    # The covariance values concerning the last symbol
    # are required to equal zero and are not represented
    # in the covariance matrix (important for taking the
    # inverse) - resulting in a matrix of size
    # (L * (num_symbols-1)) x (L * (num_symbols-1))
    # rather than (L * num_symbols) x (L * num_symbols).
    covariance_matrix = np.zeros((
        L * (num_symbols - 1),
        L * (num_symbols - 1)
    ))

    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    covariance_matrix[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols),
                    ] = f_ij[i, j, alpha, beta] - f_i[i, alpha] * f_i[j, beta]

    return covariance_matrix


@numba.jit(nopython=True)
def reshape_invC_to_4d(inv_cov_matrix, L, num_symbols):
    """
    "Un-flatten" inverse of the covariance
    matrix to allow easy access to couplings
    using position and symbol indices.

    Parameters
    ----------
    inv_cov_matrix : np.array
        The inverse of the covariance matrix.
    L : int
        Model length.
    num_symbols : int
        Number of characters in the alphabet.

    Returns
    -------
    np.array
        Matrix of size L x L x
        num_symbols x num_symbols.
    """
    J_ij = np.zeros((L, L, num_symbols, num_symbols))
    for i in range(L):
        for j in range(L):
            for alpha in range(num_symbols - 1):
                for beta in range(num_symbols - 1):
                    J_ij[i, j, alpha, beta] = inv_cov_matrix[
                        _flatten_index(i, alpha, num_symbols),
                        _flatten_index(j, beta, num_symbols)
                    ]
    return J_ij


@numba.jit(nopython=True)
def fields(J_ij, f_i):
    """
    Compute fields.

    Parameters
    ----------
    J_ij : np.array
        Matrix of size L x L x num_symbols x
        num_symbols containing coupling parameters.
    f_i : np.array
        Matrix of size L x num_symbols
        containing column frequencies.

    Returns
    -------
    np.array
        Matrix of size L x num_symbols
        containing single-site fields.
    """
    L, num_symbols = f_i.shape

    hi = np.zeros((L, num_symbols))
    for i in range(L):
        log_fi = np.log(f_i[i] / f_i[i, num_symbols - 1])
        J_ij_sum = np.zeros((1, num_symbols))
        for j in range(L):
            if i != j:
                # extract couplings relevant to position pair (i, j)
                J = J_ij[i, j]

                # some eij values over all j
                J_ij_sum += np.dot(
                    J, f_i[j].reshape((1, num_symbols)).T
                ).T

        hi[i] = log_fi - J_ij_sum

    return hi
