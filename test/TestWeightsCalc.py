import unittest
from unittest import TestCase

import numba
import numpy as np

from evcouplings.align import num_cluster_members_legacy, num_cluster_members_nogaps_parallel, map_from_alphabet, map_matrix, \
    ALPHABET_PROTEIN, MATCH_GAP, sequences_to_matrix

# Alphabet : "-" + "ACDEFGHIKLMNPQRSTVWY"

alphabet_map = map_from_alphabet(
    ALPHABET_PROTEIN, default=MATCH_GAP,
)
invalid_value = 0  # GAP character is mapped to 0


def sequences_to_mapped_matrix(sequences):
    return map_matrix(sequences_to_matrix(sequences), alphabet_map)


class TestWeights(TestCase):

    def test_helper(self, sequences, expected, identity_threshold):
        matrix = sequences_to_mapped_matrix(sequences)
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=identity_threshold, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_mapping(self):
        # Just a sanity check about the alphabet used
        seqs = ["ACD-",
                "ACD-"]
        seq_matrix = sequences_to_matrix(seqs)
        matrix_mapped = map_matrix(seq_matrix, alphabet_map)
        matrix = np.array(
            [[1, 2, 3, 0],
             [1, 2, 3, 0]]
        )
        self.assertTrue(np.array_equal(matrix_mapped, matrix), "Expected: {}\nGot: {}".format(matrix, matrix_mapped))

    def test_nogaps_basic(self):
        # Basic test
        # Symmetric and all equal
        seqs = ["ACD-",
                "ACD-",
                "ACD-"]
        matrix = sequences_to_mapped_matrix(seqs)
        expected = np.array([3, 3, 3])
        # num_neighbours = num_cluster_members(matrix, identity_threshold=0.8)  #, invalid_value=invalid_value
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.8, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected), "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Just below 0.5 sequence similarity threshold
        # Gaps not counted, so 1/3 similarity < 0.5
        seqs = ["ACD-",
                "AAA-"]
        matrix = sequences_to_mapped_matrix(seqs)
        expected = np.array([1, 1])
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Mostly gaps, so sequences are just [0] and [1]
        seqs = ["A----",
                "C----"]
        matrix = sequences_to_mapped_matrix(seqs)
        expected = np.array([1, 1])
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Not counting its own gaps makes the threshold asymmetric
        # Sequence A: only [2,3] our of 5 overlap, so 0.4 < 0.5, so sequence on its own
        # Sequence B: Only [2,3] out of 3 non-gap overlap, so 0.667 > 0.5 and seq A is a neighbour of seq B
        seqs = ["ACMMAA",
                "--MMY-"]
        matrix = sequences_to_mapped_matrix(seqs)
        # matrix = np.array(
        #     [[1, 2, 3, 4, 1, 1],
        #      [0, 0, 3, 4, 5, 0]]
        # )
        expected = np.array([1, 2])
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_num_cluster_fragment(self):
        # Problem with fragments: N sequences appear in cluster together (so num_cluster_members is N) but a fragment appears as its own cluster
        seqs = ["ACDMMMA",
                "ACDMMMY",
                "ACCMMMA",
                "ACD----"]  # Fragment
        # Note: The fragment isn't a neighbour of the full sequences (2 or 3 positions out of 7), but the full sequences are neighbours of the fragment
        expected = np.array([3, 3, 3, 4])

        matrix = sequences_to_mapped_matrix(seqs)
        # matrix = np.array(
        #     [[0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 7],
        #      [0, 1, 1, 3, 4, 5, 6],
        #      [0, 1, 2, -1, -1, -1, -1],  # Fragment
        #      ]
        # )
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_many_gaps(self):
        # Testing a non-fragment sequence with many gaps
        seqs = ["A-C-DM-",
                "A--YL-A"]
        expected = np.array([1, 1])

        matrix = sequences_to_mapped_matrix(seqs)
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_gaps_nomatch(self):
        # If gaps were counted as matches, this would be a cluster with 2 sequences (5 out of 6 matches)
        # Instead the overlap is 1/3 matches, so not above threshold
        seqs = ["ACD---",
                "AYY---"]
        expected = np.array([1, 1])

        matrix = sequences_to_mapped_matrix(seqs)
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_num_cluster_full(self):
        """TODO Test on a full MSA, running alignment etc, just to verify"""
        pass


class TestWeightsCompatibility(TestCase):
    """Test previous version of weights (slight differences)"""
    def test_num_cluster_members_basic(self):
        pass


if __name__ == '__main__':
    unittest.main()
