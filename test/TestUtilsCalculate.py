import unittest
from unittest import TestCase
from evcouplings.utils import *
from evcouplings.couplings import CouplingsModel
import numpy as np

MONO_MODEL = os.getenv('HOME') + "/evcouplings_test_cases/monomer_test/couplings/RASH_HUMAN_b03.model"
# MONO_MODEL = "/home/travis/evcouplings_test_cases/monomer_test/couplings/RASH_HUMAN_b03.model"


class TestUtilsHelpers(TestCase):

    def setUp(self):
        self.x = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    def test_entropy(self):
        """
        test entropy calculation
        """
        self.assertAlmostEqual(entropy(self.x), 1.9609640474436811)

    def test_entropy_normalized(self):
        """
        test entropy calculation with normalization
        """

        self.assertAlmostEqual(entropy(self.x, normalize=True),
                               0.15545875354128547)

    def test_entropy_map(self):
        """
        tests if generated dic contains valid entries
        """
        model = CouplingsModel(MONO_MODEL)
        entropy_dict = entropy_map(model, normalize=True)

        # must have the same length as index list
        self.assertEqual(len(entropy_dict), len(model.index_list))
        # must entropy values must be between 0 and 1
        self.assertTrue(all( 0.0 <= v <= 1.0 for v in entropy_dict.values()))
        # keys must be in index_list
        index_set = set(model.index_list)
        self.assertTrue(all(k in index_set for k in entropy_dict.keys()))

    def test_dihedral_angle(self):
        """
        test dihedral angle calculation

        note: dihedral_angle returns angles in radiant not degree
        """
        # some atom coordinates for testing
        p0 = np.array([24.969, 13.428, 30.692])  # N
        p1 = np.array([24.044, 12.661, 29.808])  # CA
        p2 = np.array([22.785, 13.482, 29.543])  # C
        p3 = np.array([21.951, 13.670, 30.431])  # O
        p4 = np.array([23.672, 11.328, 30.466])  # CB
        p5 = np.array([22.881, 10.326, 29.620])  # CG
        p6 = np.array([23.691, 9.935, 28.389])  # CD1
        p7 = np.array([22.557, 9.096, 30.459])  # CD2

        self.assertAlmostEqual(dihedral_angle(p0, p1, p2, p3), np.deg2rad(-71.21515),  places=5)
        self.assertAlmostEqual(dihedral_angle(p0, p1, p4, p5), np.deg2rad(-171.94319),  places=5)
        self.assertAlmostEqual(dihedral_angle(p1, p4, p5, p6), np.deg2rad(60.82226),  places=5)
        self.assertAlmostEqual(dihedral_angle(p1, p4, p5, p7), np.deg2rad(-177.63641),  places=5)


if __name__ == '__main__':
    unittest.main()
