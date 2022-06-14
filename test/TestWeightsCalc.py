import unittest
from unittest import TestCase

import numba
import numpy as np

from evcouplings.align import map_from_alphabet, map_matrix, ALPHABET_PROTEIN, MATCH_GAP, sequences_to_matrix
from evcouplings.couplings.weights import num_cluster_members_legacy, num_cluster_members_nogaps_parallel, num_cluster_members_nogaps_serial

# Alphabet : "-" + "ACDEFGHIKLMNPQRSTVWY"

alphabet_map = map_from_alphabet(
    ALPHABET_PROTEIN, default=MATCH_GAP,
)
invalid_value = 0  # GAP character is mapped to 0


def sequences_to_mapped_matrix(sequences):
    return map_matrix(sequences_to_matrix(sequences), alphabet_map)


class TestWeights(TestCase):

    def assert_equal(self, sequences, expected, identity_threshold):
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
        num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.8, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected), "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Just below 0.5 sequence similarity threshold
        # Gaps not counted, so 1/3 similarity < 0.6
        seqs = ["ACD-",
                "AAA-"]
        expected = np.array([1, 1])
        self.assert_equal(seqs, expected, 0.6)

        # Just above 0.6 sequence similarity threshold
        seqs = ["MMA-",
                "MMY-"]
        expected = np.array([2, 2])
        self.assert_equal(seqs, expected, 0.6)

        # Mostly gaps, so sequences are just "A" and "C"
        seqs = ["A----",
                "C----"]
        expected = np.array([1, 1])
        self.assert_equal(seqs, expected, 0.8)

        # Not counting its own gaps makes the threshold asymmetric
        # Sequence A: only "MM" our of 6 overlap, so 0.4 < 0.6, so sequence on its own
        # Sequence B: Only "MM" out of 3 non-gap MMY overlap, so 0.667 > 0.6 and seq A is a neighbour of seq B
        seqs = ["ACMMAA",
                "--MMY-"]
        # matrix = np.array(
        #     [[1, 2, 3, 4, 1, 1],
        #      [0, 0, 3, 4, 5, 0]]
        # )
        expected = np.array([1, 2])
        self.assert_equal(seqs, expected, 0.6)

    def test_num_cluster_fragment(self):
        # Problem with fragments: N sequences appear in cluster together (so num_cluster_members is N) but a fragment appears as its own cluster
        seqs = ["ACDMMMA",
                "ACDMMMY",
                "ACCMMMA",
                "ACD----"]  # Fragment
        # Note: The fragment isn't a neighbour of the full sequences (2 or 3 positions out of 7), but the full sequences are neighbours of the fragment
        expected = np.array([3, 3, 3, 4])
        # matrix = np.array(
        #     [[0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 7],
        #      [0, 1, 1, 3, 4, 5, 6],
        #      [0, 1, 2, -1, -1, -1, -1],  # Fragment
        #      ]
        # )
        self.assert_equal(seqs, expected, 0.6)

    def test_many_gaps(self):
        # Testing a non-fragment sequence with many gaps
        seqs = ["A-C-DM-",
                "A--YL-A"]
        expected = np.array([1, 1])
        self.assert_equal(seqs, expected, 0.5)

    def test_gaps_nomatch(self):
        # If gaps were counted as matches, this would be a cluster with 2 sequences (5 out of 6 matches)
        # Instead the overlap is 1/3 matches, so not above threshold
        seqs = ["ACD---",
                "AYY---"]
        expected = np.array([1, 1])

        self.assert_equal(seqs, expected, 0.5)

    def test_fragment_neighbors(self):
        # Two neighbor fragments: Similarity = 5/6 > 0.8
        seqs = ["---ACD-MMM---",
                "---ACD-MMY---"]
        expected = np.array([2, 2])
        self.assert_equal(seqs, expected, 0.8)

    def test_num_cluster_full(self):
        """TODO Test on a full MSA, running alignment etc, just to verify"""
        pass

    def test_parallel(self):
        # Numba sets the number of threads when importing, so this test won't work on a single CPU system
        # Test on a large MSA
        if numba.get_num_threads() == 1:
            print("Skipping test_parallel, as only one thread available")
            return

        try:
            numba.set_num_threads(2)  # Just use 2 to test parallelism
            np.random.seed(0)
            # 20 AAs + gaps. Note: Gap distribution will be uniform across the sequences with p=1/21,
            # which is not representative of natural sequences
            size = 1_000

            matrix = np.random.randint(0, 21, size=(size, size), dtype=int)
            serial_num_neighbours = num_cluster_members_nogaps_serial(matrix, identity_threshold=0.8, invalid_value=0)
            parallel_num_neighbours = num_cluster_members_nogaps_parallel(matrix, identity_threshold=0.8, invalid_value=0)

            self.assertTrue(np.array_equal(serial_num_neighbours, parallel_num_neighbours))
        finally:
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)


class TestWeightsLegacy(TestCase):
    """Test previous version of weights (slight differences)"""
    def assert_equal(self, sequences, expected, identity_threshold):
        matrix = sequences_to_mapped_matrix(sequences)
        num_neighbours = num_cluster_members_legacy(matrix, identity_threshold=identity_threshold)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_basic(self):
        # Repeating some of the above tests
        # Symmetric and all equal
        seqs = ["ACD-",
                "ACD-",
                "ACD-"]
        expected = np.array([3, 3, 3])
        self.assert_equal(seqs, expected, 0.8)

        # Just above 0.5 sequence similarity threshold when counting gaps
        # 3/4 similarity including gaps > 0.6
        seqs = ["AAD-",
                "AAA-"]
        expected = np.array([2, 2])
        self.assert_equal(seqs, expected, 0.6)

    def test_fragment_legacy(self):
        seqs = ["ACDMMMA",
                "ACDMMMY",
                "ACCMMMA",
                "ACD----"]  # Fragment
        # The fragment appears on its own ("ACD" matches = 2 out of 7 positions including gaps)
        expected = np.array([3, 3, 3, 1])
        # matrix = np.array(
        #     [[0, 1, 2, 3, 4, 5, 6],
        #      [0, 1, 2, 3, 4, 5, 7],
        #      [0, 1, 1, 3, 4, 5, 6],
        #      [0, 1, 2, -1, -1, -1, -1],  # Fragment
        #      ]
        # )
        self.assert_equal(seqs, expected, 0.6)


if __name__ == '__main__':
    unittest.main()
