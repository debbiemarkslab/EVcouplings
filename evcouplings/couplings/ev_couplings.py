"""
EVcouplings module v4.0

Major changes:
 * Python 3 compatibility
 * Drop legacy eij file support
 * Speed up calculations using numba
 * Clean up calculation methods

Author: Thomas A. Hopf (thomas.hopf@gmail.com)
Date: 17.04.2016
"""

# ensure Python 3 compatibility
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import range, dict, str
from six import string_types
# do not import Py3 open here - this creates a problem with numpy.fromfile since it
# does not identify as an open file object anymore

from collections import defaultdict, Iterable
from copy import deepcopy

from numba import jit
import numpy as np
import pandas as pd

# Constants

ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_PROTEIN_GAP = "-" + ALPHABET_PROTEIN_NOGAP

ALPHABET_DNA_NOGAP = "ACGT"
ALPHABET_DNA_GAP = "-" + ALPHABET_DNA_NOGAP

ALPHABET_RNA_NOGAP = "ACGU"
ALPHABET_RNA_GAP = "-" + ALPHABET_RNA_NOGAP

NUM_SYMBOLS_TO_ALPHABET = {
    4: ALPHABET_DNA_NOGAP,
    5: ALPHABET_DNA_GAP,
    20: ALPHABET_PROTEIN_NOGAP,
    21: ALPHABET_PROTEIN_GAP,
}

_SLICE = np.s_[:]
HAMILTONIAN_COMPONENTS = [FULL, COUPLINGS, FIELDS] = [0, 1, 2]
NUM_COMPONENTS = len(HAMILTONIAN_COMPONENTS)

# Methods for fast calculations (moved outside of class for numba jit)


@jit(nopython=True)
def _hamiltonians(sequences, e_ij, h_i):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from e_ij and h_i parameters

    Parameters
    ----------
    sequences : np.array
        Sequence matrix for which Hamiltonians will be computed
    e_ij: np.array
        L x L x num_symbols x num_symbols e_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Float matrix of size len(sequences) x 3, where each row corresponds to the
        1) total Hamiltonian of sequence and the 2) e_ij and 3) h_i sub-sums
    """
    # iterate over sequences
    N, L = sequences.shape
    H = np.zeros((N, NUM_COMPONENTS))
    for s in range(N):
        A = sequences[s]
        hi_sum = 0.0
        eij_sum = 0.0
        for i in range(L):
            hi_sum += h_i[i, A[i]]
            for j in range(i + 1, L):
                eij_sum += e_ij[i, j, A[i], A[j]]

        H[s] = [eij_sum + hi_sum, eij_sum, hi_sum]

    return H


@jit(nopython=True)
def _single_mutant_hamiltonians(target_seq, e_ij, h_i):
    """
    Calculate matrix of all possible single-site substitutions

    Parameters
    ----------
    L : int
        Length of model
    num_symbols : int
        Number of states of model
    target_seq : np.array(int)
        Target sequence for which mutant energy differences will be calculated
    e_ij: np.array
        L x L x num_symbols x num_symbols e_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Float matrix of size L x num_symbols x 3, where the first two dimensions correspond to
        Hamiltonian differences compared to target sequence for all possible substitutions in
        all positions, and the third dimension corresponds to the deltas of
        1) total Hamiltonian and the 2) e_ij and 3) h_i sub-sums
    """
    L, num_symbols = h_i.shape
    H = np.empty((L, num_symbols, NUM_COMPONENTS))

    # iterate over all positions
    for i in range(L):
        # iterate over all substitutions
        for A_i in range(num_symbols):
            # iterate over couplings to all other sites
            delta_hi = h_i[i, A_i] - h_i[i, target_seq[i]]
            delta_eij = 0.0

            for j in range(L):
                if i != j:
                    delta_eij += (
                        e_ij[i, j, A_i, target_seq[j]] -
                        e_ij[i, j, target_seq[i], target_seq[j]]
                    )

            H[i, A_i] = [delta_eij + delta_hi, delta_eij, delta_hi]

    return H


@jit(nopython=True)
def _delta_hamiltonian(pos, subs, target_seq, e_ij, h_i):
    """
    Parameters
    ----------
    pos : np.array(int)
        Vector of substituted positions
    subs : np.array(int)
        Vector of symbols above positions are substituted to
    target_seq : np.array(int)
        Target sequence for which mutant energy differences will be calculated
        relative to
    e_ij: np.array
        L x L x num_symbols x num_symbols e_ij pair coupling parameter matrix
    h_i: np.array
        L x num_symbols h_i fields parameter matrix

    Returns
    -------
    np.array
        Vector of length 3, where elements correspond to delta of
        1) total Hamiltonian and the 2) e_ij and 3) h_i sub-sums
    """
    L, num_symbols = h_i.shape

    M = pos.shape[0]
    delta_hi = 0.0
    delta_eij = 0.0

    # iterate over all changed positions
    for m in range(M):
        i = pos[m]
        A_i = subs[m]

        # change in fields
        delta_hi += h_i[i, A_i] - h_i[i, target_seq[i]]

        # couplings to all other positions in sequence
        for j in range(L):
            if i != j:
                delta_eij += (
                    e_ij[i, j, A_i, target_seq[j]] -
                    e_ij[i, j, target_seq[i], target_seq[j]]
                )

        # correct couplings between substituted positions:
        # 1) do not count coupling twice (remove forward
        #     and backward coupling)
        # 2) adjust background to new sequence
        for n in range(m + 1, M):
            j = pos[n]
            A_j = subs[n]
            # remove forward and backward coupling delta
            delta_eij -= e_ij[i, j, A_i, target_seq[j]]
            delta_eij -= e_ij[i, j, target_seq[i], A_j]
            delta_eij += e_ij[i, j, target_seq[i], target_seq[j]]
            # the following line cancels out with line further down:
            # delta_eij += e_ij[i, j, target_seq[i], target_seq[j]]

            # now add coupling delta once in correct background
            delta_eij += e_ij[i, j, A_i, A_j]
            # following line cancels out with line above:
            # delta_eij -= e_ij[i, j, target_seq[i], target_seq[j]]

    return np.array([delta_eij + delta_hi, delta_eij, delta_hi])

# EVcouplings class


class EVcouplings(object):
    """
    Class to store raw e_ij, h_i, P_i values from PLM parameter estimation, and allows to compute
    evolutionary couplings, sequence statistical energies, and all sorts of other things.
    """
    def __init__(self, filename, alphabet=None, precision="float32"):
        """
        Initializes the object with raw values read from binary .eij file

        Parameters
        ----------
        filename : str
            Binary eij file containing model parameters from plmc software
        alphabet : str
            Symbols corresponding to model states (e.g. "-ACGT").
        precision : {"float32", "float64"}
            Sets if input file has single (default) or double precision
        }
        """
        self.__read_plmc(filename, precision)

        # use given alphabet or guess based on number of symbols in eij file
        if alphabet is not None:
            self.alphabet = alphabet
        else:
            if self.num_symbols in NUM_SYMBOLS_TO_ALPHABET:
                self.alphabet = NUM_SYMBOLS_TO_ALPHABET[self.num_symbols]
            else:
                raise ValueError(
                    "No default alphabet with {} symbols available. "
                    "Set alphabet parameter to define. "
                    "Available defaults: {}".format(self.num_symbols, NUM_SYMBOLS_TO_ALPHABET)
                )

        self.alphabet = np.array(list(self.alphabet))
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

        # in non-gap mode, focus sequence is still coded with a gap character,
        # but gap is not part of model alphabet anymore
        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros((self.L), dtype=np.int32)
            self.has_target_seq = False

        self._reset_precomputed()

    def _reset_precomputed(self):
        """
        Delete precomputed values (e.g. mutation matrices)
        """
        self.single_mut_mat = None
        self.single_mut_mat_full = None
        self.double_mut_mat = None
        self.cn_scores = None
        self.fn_scores = None

    def __read_plmc(self, filename, precision):
        """
        Read updated eij file format from plmc.

        Parameters
        ----------
        filename : str
            Binary eij file containing model parameters
        precision : {"float32", "float64"}
            Sets if input file has single (default) or double precision

        """
        with open(filename, "rb", buffering=0) as eij_file:

            def _read_Nxq():
                """
                Read binary N x q block matrix (f, h)
                """
                return np.fromfile(
                    eij_file, precision, self.L * self.num_symbols
                ).reshape(self.L, self.num_symbols)

            def _read_qxq():
                """
                Read binary q x q block matrix (f_ij, e_ij)
                """
                return np.fromfile(
                    eij_file, precision, self.num_symbols * self.num_symbols
                ).reshape(self.num_symbols, self.num_symbols)

            # model length, number of symbols, target sequence and index mapping
            self.L, = np.fromfile(eij_file, "int32", 1)
            self.num_symbols, = np.fromfile(eij_file, "int32", 1)
            self.target_seq = np.fromfile(eij_file, "|S1", self.L)
            self.index_list = np.fromfile(eij_file, "int32", self.L)

            # single site frequencies f_i and fields h_i
            self.f_i = _read_Nxq()
            self.h_i = _read_Nxq()

            # pair frequencies f_ij and pair couplings e_ij / J_ij
            self.f_ij = np.zeros((self.L, self.L, self.num_symbols, self.num_symbols))
            self.e_ij = np.zeros((self.L, self.L, self.num_symbols, self.num_symbols))
            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    # check consistency of indices in file with what they should be
                    file_i, file_j = np.fromfile(eij_file, "int32", 2)
                    if i + 1 != file_i or j + 1 != file_j:
                        raise ValueError(
                            "Error: column pair indices inconsistent. "
                            "Expected: {} {}; File: {} {}".format(i + 1, j + 1, file_i, file_j)
                        )

                    self.f_ij[i, j] = _read_qxq()
                    self.f_ij[j, i] = self.f_ij[i, j].T

                    self.e_ij[i, j] = _read_qxq()
                    self.e_ij[j, i] = self.e_ij[i, j].T

    def set_target_sequence(self, sequence):
        """
        Define a new target sequence

        Parameters
        ----------
        sequence : str, or list of chars
            Define a new default sequence for relative Hamiltonian
            calculations (e.g. energy difference relative to wild-type
            sequence).
            Length of sequence must correspond to model length (self.L)
        """
        # Since single and double mutant matrices are relative to target sequence
        # delete any precomputed values
        self._reset_precomputed()

        if len(sequence) != self.L:
            raise ValueError(
                "Sequence length inconsistent with model length: {} {}".format(
                    len(sequence), self.L
                )
            )

        # use six.string_types because py3 does not know "basestring" or
        # "unicode", while "str" misses unicode strings in py2
        if isinstance(sequence, string_types):
            self.target_seq = list(sequence)

        self.target_seq = np.array(self.target_seq)
        self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
        self.has_target_seq = True

    def set_index_mapping(self, mapping):
        """
        Define a new number mapping for sequences

        Parameters
        ----------
        index_map: list of int
            Sequence indices of the positions in the model.
            Length of list must correspond to model length (self.L)
        """
        if len(mapping) != self.L:
            raise ValueError(
                "Mapping length inconsistent with model length: {} {}".format(
                    len(mapping), self.L
                )
            )

        self.index_list = np.array(mapping)
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

    def convert_sequences(self, sequences):
        """
        Converts sequences in string format into internal symbol representation
        according to alphabet of model

        Parameters
        ----------
        sequences : list of str
            List of sequences (must have same length and correspond to
            model states)

        Returns
        -------
        np.array
            Matrix of size len(sequences) x L of sequences converted to
            integer symbols
        """
        seq_lens = list(set(map(len, sequences)))
        if len(seq_lens) != 1:
            raise(ValueError("Input sequences have different lengths: " + str(seq_lens)))

        L_seq = seq_lens[0]
        if L_seq != self.L:
            raise(
                ValueError("Sequence lengths do not correspond to model length: {} {}".format(
                    L_seq, self.L)
                )
            )

        S = np.empty((len(sequences), L_seq), dtype=np.int)

        try:
            for i, s in enumerate(sequences):
                S[i] = [self.alphabet_map[x] for x in s]
        except KeyError:
            raise ValueError("Invalid symbol in sequence {}: {}".format(i, x))

        return S

    def hamiltonians(self, sequences):
        """
        Calculates the Hamiltonians of the global probability distribution P(A_1, ..., A_L)
        for the given sequences A_1,...,A_L from e_ij and h_i parameters

        Parameters
        ----------
        sequences : list of str
            List of sequences for which Hamiltonian will be computed,
            or converted np.array obtained using convert_sequences method

        Returns
        -------
        np.array
            Float matrix of size len(sequences) x 3, where each row corresponds to the
            1) total Hamiltonian of sequence and the 2) e_ij and 3) h_i sub-sums
        """
        if isinstance(sequences, list):
            sequences = self.convert_sequences(sequences)

        return _hamiltonians(sequences, self.e_ij, self.h_i)

    def calculate_single_mutants(self):
        """
        Calculates Hamiltonian difference for all possible single-site variants

        Returns
        -------
        np.array(float)
            L x num_symbol x 3 matrix containing delta Hamiltonians
            for all possible single mutants of target sequence.
            Third dimension: 1) full Hamiltonian, 2) e_ij, 3) h_i
        """
        self.single_mut_mat_full = _single_mutant_hamiltonians(
            self.target_seq_mapped, self.e_ij, self.h_i
        )

        self.single_mut_mat = self.single_mut_mat_full[:, :, FULL]

        return self.single_mut_mat_full

    def delta_hamiltonian(self, substitutions, verify_mutants=True):
        """
        Calculate difference in statistical energy relative to
        self.target_seq by changing sequence according to list of
        substitutions

        Parameters
        ----------
        substitutions : list of tuple(pos, subs_from, subs_to)
            Substitutions to be applied to target sequence
        verify_mutants : bool, optional
            Test if subs_from is consistent with self.target_seq

        Returns
        -------
        np.array
            Vector of length 3 with 1) total delta Hamiltonian,
            2) delta e_ij, 3) delta h_i

        """
        pos = np.empty(len(substitutions), dtype=np.int)
        subs = np.empty(len(substitutions), dtype=np.int)

        try:
            for i, (subs_pos, subs_from, subs_to) in enumerate(substitutions):
                pos[i] = self.index_map[subs_pos]
                subs[i] = self.alphabet_map[subs_to]
                if verify_mutants and subs_from != self.target_seq[pos[i]]:
                    raise ValueError(
                        "Inconsistency with target sequence: pos={} target={} subs={}".format(
                            subs_pos, self.target_seq[i], subs_from
                        )
                    )
        except KeyError:
            raise ValueError(
                "Illegal substitution: {}{}{}\nAlphabet: {}\nPositions: {}".format(
                    subs_from, subs_pos, subs_to, self.alphabet_map, self.index_list
                )
            )

        return _delta_hamiltonian(pos, subs, self.target_seq_mapped, self.e_ij, self.h_i)

    def calculate_double_mutants(self):
        """
        Calculates Hamiltonian difference for all possible double-site variants, using
        information calculated for single-site variants

        Returns
        -------
        np.array(float)
            L x L x num_symbol x num_symbol matrix containing delta Hamiltonians
            for all possible double mutants of target sequence
        """
        if self.single_mut_mat is None:
            self.calculate_single_mutants()

        self.double_mut_mat = np.zeros(
            (self.L, self.L, self.num_symbols, self.num_symbols)
        )

        seq = self.target_seq_mapped
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                self.double_mut_mat[i, j] = (
                    np.tile(self.single_mut_mat[i], (self.num_symbols, 1)).T +
                    np.tile(self.single_mut_mat[j], (self.num_symbols, 1)) +
                    self.e_ij[i, j] -
                    np.tile(self.e_ij[i, j, :, seq[j]], (self.num_symbols, 1)).T -
                    np.tile(self.e_ij[i, j, seq[i], :], (self.num_symbols, 1)) +
                    # we are only interested in difference to WT, so normalize
                    # for second couplings subtraction with last term
                    self.e_ij[i, j, seq[i], seq[j]])

                self.double_mut_mat[j, i] = self.double_mut_mat[i, j].T

        return self.double_mut_mat

    def __apc(self, matrix):
        """
        Apply average product correction (Dunn et al., Bioinformatics, 2008)
        to matrix

        Parameters
        ----------
        matrix : np.array
            Symmetric L x L matrix which should be corrected by APC

        Returns
        -------
        np.array
            Symmetric L x L matrix with APC correction applied
        """
        L = matrix.shape[0]
        if L != matrix.shape[1]:
            raise ValueError("Input matrix is not symmetric: {}".format(matrix.shape))

        col_means = np.mean(matrix, axis=0) * L / (L - 1)
        matrix_mean = np.mean(matrix) * L / (L - 1)

        apc = np.dot(
            col_means.reshape(L, 1), col_means.reshape(1, L)
        ) / matrix_mean

        # subtract APC and blank diagonal entries
        corrected_matrix = matrix - apc
        corrected_matrix[np.diag_indices(L)] = 0

        return corrected_matrix

    def calculate_ecs(self):
        """
        Calculates FN and CN scores as defined in Ekeberg et al., Phys Rev E, 2013,
        as well as MI scores. Assumes parameters are in zero-sum gauge.

        Set member variables
        --------------------
        self.fn_scores : np.array(float)
            L x L matrix with FN scores
        self.cn_scores : np.array(float)
            L x L matrix with CN scores
        self.mi_scores_raw : np.array(float)
            L x L matrix with MI scores (no APC correction)
        self.mi_scores_apc : np.array(float)
            L x L matrix with MI scores (with APC correction)
        self.ec_list: pd.DataFrame
            Dataframe with computed evolutionary couplings

        Returns
        -------
        pd.DataFrame
            Dataframe with computed ECs for all pairs of positions
        """
        # calculate Frobenius norm for each pair of sites (i, j)
        # also calculate mutual information
        self.fn_scores = np.zeros((self.L, self.L))
        self.mi_scores_raw = np.zeros((self.L, self.L))

        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                self.fn_scores[i, j] = np.linalg.norm(self.e_ij[i, j], "fro")
                self.fn_scores[j, i] = self.fn_scores[i, j]

                # mutual information
                p = self.f_ij[i, j]
                m = np.dot(self.f_i[i, np.newaxis].T, self.f_i[j, np.newaxis])
                self.mi_scores_raw[i, j] = np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
                self.mi_scores_raw[j, i] = self.mi_scores_raw[i, j]

        # apply Average Product Correction (Dunn et al., Bioinformatics, 2008)
        # subtract APC and blank diagonal entries
        self.cn_scores = self.__apc(self.fn_scores)
        self.mi_scores_apc = self.__apc(self.mi_scores_raw)

        # create internal dataframe representation
        ecs = []
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                ecs.append((
                    self.index_list[i], self.target_seq[i],
                    self.index_list[j], self.target_seq[j],
                    abs(self.index_list[i] - self.index_list[j]),
                    self.mi_scores_raw[i, j], self.mi_scores_apc[i, j],
                    self.fn_scores[i, j], self.cn_scores[i, j]
                ))

        self.ec_list = pd.DataFrame(
            ecs, columns=["i", "A_i", "j", "A_j", "seqdist", "mi_raw", "mi_apc", "fn", "cn"]
        ).sort_values(by="cn", ascending=False)

        return self.ec_list

    def to_independent_model(self, N, lambda_h=0.01):
        """
        Estimate parameters of a single-site model using
        Gaussian prior/L2 regularization.

        Parameters
        ----------
        N : float
            Effective (reweighted) number of sequences
        lambda_h : float
            Strength of L2 regularization on h_i parameters

        Returns
        -------
        EVcouplings
            Copy of object turned into independent model
        """
        from scipy.optimize import fmin_bfgs

        def _log_post(x, *args):
            """
            Log posterior of single-site model
            """
            (fi, lambda_h, N) = args
            logZ = np.log(np.exp(x).sum())
            return N * (logZ - (fi * x).sum()) + lambda_h * ((x**2).sum())

        def _gradient(x, *args):
            """
            Gradient of single-site model
            """
            (fi, lambda_h, N) = args
            Z = np.exp(x).sum()
            P = np.exp(x) / Z
            return N * (P - fi) + lambda_h * 2 * x

        h_i = np.zeros((self.L, self.num_symbols))

        for i in range(self.L):
            x0 = np.zeros((self.num_symbols))
            h_i[i] = fmin_bfgs(
                _log_post, x0, _gradient,
                args=(self.f_i[i], lambda_h, N), disp=False
            )

        c0 = deepcopy(self)
        c0.h_i = h_i
        c0.e_ij.fill(0)
        c0._reset_precomputed()
        return c0

    # syntactic sugar to access most important member variables in target numbering space

    def __map(self, indices, mapping):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices : Iterable of items to be mapped, or single item
        mapping: Dictionary containing mapping into new space

        Returns
        -------
        Iterable, or single item
            Items mapped into new space
        """
        if ((isinstance(indices, Iterable) and not isinstance(indices, string_types)) or
                (isinstance(indices, string_types) and len(indices) > 1)):
            return np.array(map(lambda x: mapping[x], indices))
        else:
            return mapping[indices]

    def __4d_access(self, matrix, i=None, j=None, A_i=None, A_j=None):
        """
        Provides shortcut access to column pair properties
        (e.g. e_ij or f_ij matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        j : Iterable(int) or int
            Position(s) on second matrix axis
        A_i : Iterable(str) or str
            Symbols corresponding to first matrix axis
        A_j : Iterable(str) or str
            Symbols corresponding to second matrix axis

        Returns
        -------
        np.array
            4D matrix "matrix" sliced according to values i, j, A_i and A_j
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        j = self.__map(j, self.index_map) if j is not None else _SLICE
        A_i = self.__map(A_i, self.alphabet_map) if A_i is not None else _SLICE
        A_j = self.__map(A_j, self.alphabet_map) if A_j is not None else _SLICE
        return matrix[i, j, A_i, A_j]

    def __2d_access(self, matrix, i=None, A_i=None):
        """
        Provides shortcut access to single-column properties
        (e.g. f_i or h_i matrices)

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        A_i : Iterable(str) or str
            Symbols corresponding to first matrix axis

        Returns
        -------
        np.array
            2D matrix "matrix" sliced according to values i and A_i
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        A_i = self.__map(A_i, self.alphabet_map) if A_i is not None else _SLICE
        return matrix[i, A_i]

    def __2d_access_score_matrix(self, matrix, i=None, j=None):
        """
        Provides shortcut access to quadratic 2D matrices

        Parameters
        -----------
        i : Iterable(int) or int
            Position(s) on first matrix axis
        j : Iterable(int) or int
            Position(s) on first matrix axis

        Returns
        -------
        np.array
            2D matrix "matrix" sliced according to values i and j
        """
        i = self.__map(i, self.index_map) if i is not None else _SLICE
        j = self.__map(j, self.index_map) if j is not None else _SLICE
        return matrix[i, j]

    def eij(self, i=None, j=None, A_i=None, A_j=None):
        """
        Quick access to e_ij matrix with automatic index mapping.
        See __4d_access for explanation of parameters.
        """
        return self.__4d_access(self.e_ij, i, j, A_i, A_j)

    def fij(self, i=None, j=None, A_i=None, A_j=None):
        """
        Quick access to f_ij matrix with automatic index mapping.
        See __4d_access for explanation of parameters.
        """
        return self.__4d_access(self.f_ij, i, j, A_i, A_j)

    def hi(self, i=None, A_i=None):
        """
        Quick access to h_i matrix with automatic index mapping.
        See __2d_access for explanation of parameters.
        """
        return self.__2d_access(self.h_i, i, A_i)

    def fi(self, i=None, A_i=None):
        """
        Quick access to f_i matrix with automatic index mapping.
        See __2d_access for explanation of parameters.
        """
        return self.__2d_access(self.f_i, i, A_i)

    def cn(self, i=None, j=None):
        """
        Quick access to cn_scores matrix with automatic index mapping.
        See __2d_access_score_matrix for explanation of parameters.
        """
        if self.cn_scores is None:
            self.calculate_ecs()
        return self.__2d_access_score_matrix(self.cn_scores, i, j)

    def fn(self, i=None, j=None):
        """
        Quick access to fn_scores matrix with automatic index mapping.
        See __2d_access_score_matrix for explanation of parameters.
        """
        if self.fn_scores is None:
            self.calculate_ecs()
        return self.__2d_access_score_matrix(self.fn_scores, i, j)

    def mi_apc(self, i=None, j=None):
        """
        Quick access to mi_scores_apc matrix with automatic index mapping.
        See __2d_access_score_matrix for explanation of parameters.
        """
        if self.mi_scores_apc is None:
            self.calculate_ecs()
        return self.__2d_access_score_matrix(self.mi_scores_apc, i, j)

    def mi_raw(self, i=None, j=None):
        """
        Quick access to mi_scores_raw matrix with automatic index mapping.
        See __2d_access_score_matrix for explanation of parameters.
        """
        if self.mi_scores_raw is None:
            self.calculate_ecs()
        return self.__2d_access_score_matrix(self.mi_scores_raw, i, j)

    def mn(self, i=None):
        """
        Map model numbering to internal numbering

        Parameters
        ----------
        i : Iterable(int) or int
            Position(s) to be mapped from model numbering space
            into internal numbering space

        Returns
        -------
        Iterable(int) or int
            Remapped position(s)
        """
        if i is None:
            return np.array(sorted(self.index_map.values()))
        else:
            return self.__map(i, self.index_map)

    def mui(self, i=None):
        """
        Legacy method for backwards compatibility. See self.mn for explanation.
        """
        return self.mn(i)

    def tn(self, i=None):
        """
        Map internal numbering to model numbering

        Parameters
        ----------
        i : Iterable(int) or int
            Position(s) to be mapped from internal numbering space
            into model numbering space.

        Returns
        -------
        Iterable(int) or int
            Remapped position(s)
        """
        if i is None:
            return np.array(self.index_list)
        else:
            return self.__map(i, self.index_list)

    def itu(self, i=None):
        """
        Legacy method for backwards compatibility. See self.tn for explanation.
        """
        return self.tn(i)

    def seq(self, i=None):
        """
        Access target sequence of model

        Parameters
        ----------
        i : Iterable(int) or int
            Position(s) for which symbol should be retrieved

        Returns
        -------
        Iterable(char) or char
            Sequence symbols
        """
        if i is None:
            return self.target_seq
        else:
            i = self.__map(i, self.index_map)
            return self.__map(i, self.target_seq)

    def smm(self, i=None, A_i=None):
        """
        Access delta_Hamiltonian matrix of single mutants of target sequence

        Parameters
        ----------
        i : Iterable(int) or int
            Position(s) for which energy difference should be retrieved
        A_i : Iterable(char) or char
            Substitutions for which energy difference should be retrieved

        Returns
        -------
        np.array(float)
            2D matrix containing energy differences for slices along both
            axes of single mutation matrix (first axis: position, second
            axis: substitution).
        """
        if self.single_mut_mat is None:
            self.calculate_single_mutants()
        return self.__2d_access(self.single_mut_mat, i, A_i)

    def dmm(self, i=None, j=None, A_i=None, A_j=None):
        """
        Access delta_Hamiltonian matrix of double mutants of target sequence

        Parameters
        ----------
        i : Iterable(int) or int
            Position(s) of first substitution(s)
        j : Iterable(int) or int
            Position(s) of second substitution(s)
        A_i : Iterable(char) or char
            Substitution(s) to first position
        A_j : Iterable(char) or char
            Substitution(s) to second position

        Returns
        -------
        np.array(float)
            4D matrix containing energy differences for slices along both
            axes of double mutation matrix (axes 1/2: position, axis 3/4:
            substitutions).
        """

        if self.double_mut_mat is None:
            self.calculate_double_mutants()
        return self.__4d_access(self.double_mut_mat, i, j, A_i, A_j)


def load_eijs(eij_file, N, alphabet=None, lambda_h=0.01, precision="float32"):
    """
    Syntactic sugar for loading eij file and calculating
    corresponding (l2-regularized) independent model

    Parameters
    ----------
    eij_file: str
        x
    N : float
        Effective number of samples used to esimate model in eij_file
    alphabet : str
        Symbols corresponding to model states (e.g. "-ACGT").
    lambda_h : float
        Regularization strength on h_i (fields) for independent model
        estimation
    precision : {"float32", "float64"}
        Sets if input file has single (default) or double precision

    Returns
    -------
    c : EVcouplings
        Pairwise model stored in eij_file
    c0 : EVcouplings
        Corresponding independent model estimated based
        on single-site frequencies in eij_file
    """
    c = EVcouplings(eij_file, alphabet, precision)
    c.calculate_ecs()
    c0 = c.to_independent_model(N=N, lambda_h=lambda_h)

    return c, c0


class ComplexIndexMapper():
    """
    Map indices of sequences into concatenated EVcouplings
    object numbering space. Can in principle also be used
    to remap indices for a single sequence.
    """
    def __init__(self, couplings, couplings_range, *monomer_ranges):
        """
        Ranges are tuples of form (start: int, end: int)
        couplings_range must match the range of EVcouplings object
        Example: ComplexIndexMapper(c, (1, 196), (1, 103), (1, 93))

        Parameters
        ----------
        couplings : EVcouplings
            Couplings object of complex
        couplings_range : (int, int)
            Numbering range in couplings that monomers will be
            mapped to
        *monomer_ranges: (int, int):
            Tuples containing numbering range of each monomer
        """
        if len(monomer_ranges) < 1:
            raise ValueError("Give at least one monomer range")

        self.couplings_range = couplings_range
        self.monomer_ranges = monomer_ranges
        self.monomer_to_full_range = {}

        # create a list of positions per region that directly
        # aligns against the full complex range in c_range
        r_map = []
        for i, (r_start, r_end) in enumerate(monomer_ranges):
            m_range = range(r_start, r_end + 1)
            r_map += zip([i] * len(m_range), m_range)
            self.monomer_to_full_range[i] = m_range

        c_range = range(couplings_range[0], couplings_range[1] + 1)
        if len(r_map) != len(c_range):
            raise ValueError(
                "Complex range and monomer ranges do not have equivalent lengths "
                "(complex: {}, sum of monomers: {}).".format(len(c_range), len(r_map))
            )

        # These dicts might contain indices not contained in
        # couplings object because they are lowercase in alignment
        self.monomer_to_couplings = dict(zip(r_map, c_range))
        self.couplings_to_monomer = dict(zip(c_range, r_map))

        # store all indices per subunit that are actually
        # contained in couplings object
        self.monomer_indices = defaultdict(list)
        for (monomer, m_res), c_res in sorted(self.monomer_to_couplings.items()):
            if c_res in couplings.tn():
                self.monomer_indices[monomer].append(m_res)

    def __map(self, indices, mapping_dict):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices: int, or (int, int), or lists thereof
            Indices in input numbering space

        mapping_dict : dict(int->(int, int)) or dict((int, int): int)
            Mapping from one numbering space into the other

        Returns
        -------
        list of int, or list of (int, int)
            Mapped indices
        """
        if isinstance(indices, Iterable) and not isinstance(indices, tuple):
            return np.array([mapping_dict[x] for x in indices])
        else:
            return mapping_dict[indices]

    def __call__(self, monomer, res):
        """
        Function-style syntax for single residue to be mapped
        (calls toc method)

        Parameters
        ----------
        monomer : int
            Number of monomer
        res : int
            Position in monomer numbering

        Returns
        -------
        int
            Index in coupling object numbering space
        """
        return self.toc((monomer, res))

    def tom(self, x):
        """
        Map couplings TO *M*onomer

        Parameters
        ----------
        x : int, or list of ints
            Indices in coupling object

        Returns
        -------
        (int, int), or list of (int, int)
            Indices mapped into monomer numbering. Tuples are
            (monomer, index in monomer sequence)
        """
        return self.__map(x, self.couplings_to_monomer)

    def toc(self, x):
        """
        Map monomer TO *C*ouplings / complex

        Parameters
        ----------
        x : (int, int), or list of (int, int)
            Indices in momnomers (monomer, index in monomer sequence)

        Returns
        -------
        int, or list of int
            Monomer indices mapped into couplings object numbering
        """
        return self.__map(x, self.monomer_to_couplings)
