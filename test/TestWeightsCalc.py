import unittest
from unittest import TestCase

import numpy as np

from evcouplings.align import num_cluster_members_legacy, num_cluster_members_parallel


class TestWeights(TestCase):

    def test_num_cluster_members_basic(self):
        # Basic test
        invalid_value = -1  # Usually the matrix has to be between (0, num_symbols) but here we can do what we want

        # Symmetric and all equal
        matrix = np.array(
            [[0, 1, 2, -1],
             [0, 1, 2, -1],
             [0, 1, 2, -1]]
        )
        expected = np.array([3, 3, 3])
        # num_neighbours = num_cluster_members(matrix, identity_threshold=0.8)  #, invalid_value=invalid_value
        num_neighbours = num_cluster_members_parallel(matrix, identity_threshold=0.8, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected), "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Just below 0.5 sequence similarity threshold
        # Gaps not counted, so 1/3 similarity < 0.5
        matrix = np.array(
            [[0, 1, 2, -1],
             [0, 0, 0, -1]]
        )
        expected = np.array([1, 1])
        num_neighbours = num_cluster_members_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Mostly gaps, so sequences are just [0] and [1]
        matrix = np.array(
            [[0, -1, -1, -1, -1],
             [1, -1, -1, -1, -1]]
        )
        expected = np.array([1, 1])
        num_neighbours = num_cluster_members_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

        # Not counting its own gaps makes the threshold asymmetric
        # Sequence A: only [2,3] our of 5 overlap, so 0.4 < 0.5, so sequence on its own
        # Sequence B: Only [2,3] out of 3 non-gap overlap, so 0.667 > 0.5 and seq A is a neighbour of seq B
        matrix = np.array(
            [[0, 1, 2, 3, 0, 0],
             [-1, -1, 2, 3, 4, -1]]
        )
        expected = np.array([1, 2])
        num_neighbours = num_cluster_members_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
        self.assertTrue(np.array_equal(num_neighbours, expected),
                        "Expected: {}\nGot: {}".format(expected, num_neighbours))

    def test_num_cluster_fragment(self):
        invalid_value = -1  # Usually the matrix has to be between (0, num_symbols) but here we can do what we want
        # Problem: N sequences appear in cluster together (so num_cluster_members is N) but a fragment appears as its own cluster

        matrix = np.array(
            [[0, 1, 2, 3, 4, 5, 6],
             [0, 1, 2, 3, 4, 5, 7],
             [0, 1, 1, 3, 4, 5, 6],
             [0, 1, 2, -1, -1, -1, -1],  # Fragment
             ]
        )
        # Note: The fragment isn't a neighbour of the full sequences, but the full sequences are neighbours of the fragment
        expected = np.array([3, 3, 3, 4])
        num_neighbours = num_cluster_members_parallel(matrix, identity_threshold=0.5, invalid_value=invalid_value)
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
