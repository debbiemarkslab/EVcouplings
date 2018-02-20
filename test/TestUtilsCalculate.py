import os
import unittest
import numpy as np
from unittest import TestCase
from evcouplings.utils import *


class TestUtilsHelpers(TestCase):

    def setUp(self):
        self.x = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    def test_entropy(self):
        """
        test entropy calculation
        """
        self.assertAlmostEqual(entropy(self.x), 1.9609640474436811)

    def test_entropy_normalize(self):
        """
        test entropy calculation with normalization
        """

        self.assertAlmostEqual(entropy(self.x, normalize=True),
                               0.15545875354128547)


if __name__ == '__main__':
    unittest.main()