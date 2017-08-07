"""
Inference of evolutionary couplings
from sequence alignments using
mean field approximation.

Authors:
  Sophia F. Mersmann
"""

import numpy as np

from evcouplings.couplings import CouplingsModel


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
    _segment : Segment
        Segment that definces the mapping from position
        indices to UniProt space.
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
    def __init__(self, alignment, segment):
        """
        Initialize direct couplings analysis by
        processing the given alignment.

        Parameters
        ----------
        alignment : Alignment
            Alignment with lower / upper columns
            and the target sequence as first record.
        segment : Segment
            The segment used to map the index to
            UniProt space.
        """
        # input alignment and segment
        self._raw_alignment = alignment
        self._segment = segment

        # prepare alignment for DCA
        # by removing invalid sequences
        # and selecting focus columns
        self._prepare_alignment()

        # for convenience
        self.N = self.alignment.N
        self.L = self.alignment.L
        self.num_symbols = self.alignment.num_symbols

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
        self.alignment.corrected_frequencies = None
        self.alignment.corrected_pair_frequencies = None

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

        # compute column frequencies corrected by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.alignment.compute_corrected_frequencies(
            pseudo_count=pseudo_count
        )

        # compute pairwise frequencies corrected by a pseudo-count
        # (this implicitly calculates the raw frequencies as well)
        self.alignment.compute_corrected_pair_frequencies(
            pseudo_count=pseudo_count
        )

        # compute the covariance matrix from
        # the column and pair frequencies
        self.compute_covariance_matrix()

        # coupling parameters are inferred
        # by inverting the covariance matrix
        self.covariance_matrix_inv = np.linalg.inv(
            self.covariance_matrix
        )

        # reshape the inverse of the covariance matrix
        # to make eijs easily accessible
        J_ij = self.reshape_invC_to_4d()

        # compute fields
        h_i = self.fields()

        return MeanFieldCouplingsModel(
            alignment=self.alignment,
            segment=self._segment,
            h_i=h_i,
            J_ij=J_ij,
            theta=theta,
            pseudo_count=pseudo_count
        )

    def _prepare_alignment(self):
        """
        Prepare the input a2m alignment for
        mean field direct coupling analysis.

        The method processes the input alignment
        to an alignment of only valid sequences
        and focus columns. It sets the attribute
        self.alignment and returns a reference to it.

        Returns
        -------
            Reference to self.alignment
        """
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

        return self.alignment

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

        # for convenience
        fi = self.alignment.corrected_frequencies
        fij = self.alignment.corrected_pair_frequencies

        for i in range(self.L):
            for j in range(self.L):
                for alpha in range(self.num_symbols - 1):
                    for beta in range(self.num_symbols - 1):
                        self.covariance_matrix[
                            _flatten_index(i, alpha, self.num_symbols),
                            _flatten_index(j, beta, self.num_symbols),
                        ] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j, beta]

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
        J_ij = np.zeros((self.L, self.L, self.num_symbols, self.num_symbols))
        for i in range(self.L):
            for j in range(self.L):
                for alpha in range(self.num_symbols - 1):
                    for beta in range(self.num_symbols - 1):
                        J_ij[i, j, alpha, beta] = self.covariance_matrix_inv[
                            _flatten_index(i, alpha, self.num_symbols),
                            _flatten_index(j, beta, self.num_symbols)
                        ]
        return J_ij

    def fields(self):
        """
        Compute fields.

        Returns
        -------
        np.array
            Matrix of size L x num_symbols
            containing single-site fields.
        """
        # make couplings easily accessible
        J_ij = self.reshape_invC_to_4d()

        # for convenience
        fi = self.alignment.corrected_frequencies

        hi = np.zeros((self.L, self.num_symbols))
        for i in range(self.L):
            log_fi = np.log(fi[i] / fi[i, self.num_symbols - 1])
            J_ij_sum = np.zeros((1, self.num_symbols))
            for j in range(self.L):
                if i != j:
                    # extract couplings relevant to position pair (i, j)
                    J = -J_ij[i, j]

                    # some eij values over all j
                    J_ij_sum += np.dot(
                        J, fi[j].reshape((1, self.num_symbols)).T
                    ).T
            hi[i] = log_fi - J_ij_sum

        return hi


class MeanFieldCouplingsModel(CouplingsModel):
    """
    Mean field DCA specific model class that stores the
    parameters inferred using mean field approximation
    and calculates mutual and direct information as
    well as fn and cn scores.
    """
    def __init__(self, alignment, segment, h_i, J_ij,
                 theta, pseudo_count):
        """
        Initialize the model with the results of a
        mean field inference.

        Parameters
        ----------
        alignment : Alignment
            The alignment that was used inferring
            model parameters using mean field approximation.
        segment : Segment
            The segment is used for mapping indices
            into UniProt space.
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

        # set plmc-specific parameters to arbitrary value
        # (cannot be set to None, but must be set to a
        # numerical value for the to_file method to work properly)
        self.num_iter = -1
        self.lambda_h = -1
        self.lambda_J = -1
        self.lambda_group = -1

        # raw single and pair frequencies
        self.f_i = alignment.frequencies
        self.f_ij = alignment.pair_frequencies

        # corrected single and pair frequencies
        self.corrected_f_i = alignment.corrected_frequencies
        self.corrected_f_ij = alignment.corrected_pair_frequencies

        # fields and couplings
        self.h_i = h_i
        self.J_ij = J_ij

        # parameters used in mean field inference
        self.theta = theta
        self.pseudo_count = pseudo_count

        # mapping
        self.index_list = np.array(segment.positions)

        # for now, the number of invalid sequences
        # is set to zero since __read_plmc_v2() in
        # couplings.model reads in the weights of
        # valid and invalid sequences but here only
        # weights of the valid sequences are used -
        # so, the number of invalid sequences must be
        # zero to prevent reading the model file
        # from crashing
        self.N_invalid = 0

        self._reset_precomputed()

    @classmethod
    def from_file(cls, filename, precision="float32", file_format="plmc_v2", **kwargs):
        """
        Read the model from a binary .Jij file

        For a detailed description of the method's parameters,
        have a look at the __init__ method of
        evcouplings.couplings.model.CouplingsModel.
        """
        super().__init__(filename, precision, file_format, **kwargs)

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
        self._di_scores = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                # extract couplings relevant to
                # position pair (i, j)
                J = np.exp(-self.J_ij[i, j])

                # compute two-site model
                h_tilde_i, h_tilde_j = self.tilde_fields(i, j)
                p_di_ij = J * np.dot(h_tilde_i.T, h_tilde_j)
                p_di_ij = p_di_ij / p_di_ij.sum()

                # dot product of single-site frequencies
                # of columns i and j
                f_ij = np.dot(
                    self.corrected_f_i[i].reshape((1, self.num_symbols)).T,
                    self.corrected_f_i[j].reshape((1, self.num_symbols))
                )

                # finally, compute direct information as
                # mutual information associated to p_di_ij
                _TINY = 1.0e-100
                self._di_scores[i, j] = np.trace(
                    np.dot(
                        p_di_ij.T,
                        np.log((p_di_ij + _TINY) / (f_ij + _TINY)))
                )
                self._di_scores[j, i] = self._di_scores[i, j]

        # add DI scores to EC data frame
        di = []
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                di.append(self._di_scores[i, j])

        self._ecs.sort_values(by=["i", "j"], inplace=True)
        self._ecs["di"] = di

        return self._ecs.sort_values(
            by="di", ascending=False
        )

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
        _EPSILON = 1e-4
        diff = 1.0

        h_tilde_i = np.full((1, self.num_symbols), 1 / float(self.num_symbols))
        h_tilde_j = np.full((1, self.num_symbols), 1 / float(self.num_symbols))

        J = np.exp(-self.J_ij[i, j])
        while diff > _EPSILON:
            tmp_1 = np.dot(h_tilde_j, J.T)
            tmp_2 = np.dot(h_tilde_i, J)

            h_tilde_i_updated = self.corrected_f_i[i] / tmp_1
            h_tilde_i_updated /= h_tilde_i_updated.sum()

            h_tilde_j_updated = self.corrected_f_i[j] / tmp_2
            h_tilde_j_updated /= h_tilde_j_updated.sum()

            diff = max(
                np.absolute(h_tilde_i_updated - h_tilde_i).max(),
                np.absolute(h_tilde_j_updated - h_tilde_j).max()
            )

            h_tilde_i = h_tilde_i_updated
            h_tilde_j = h_tilde_j_updated

        return h_tilde_i, h_tilde_j

    @property
    def di_scores(self):
        """
        L x L numpy matrix with DI (direct information) scores
        """
        if self._di_scores is None:
            self._calculate_ecs()

        return self._di_scores

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
                        "{0:.6f}".format(self.di_scores[i, j])
                    ])) + "\n")
