"""
Test cases for couplings stage of EVCouplingspipeline

Author:
    Anna G. Green
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest import TestCase
from evcouplings.couplings.pairs import *

# TRAVIS_PATH = "/home/travis/evcouplings_test_cases/complex_test"
TRAVIS_PATH = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/complex_test"
TRAVIS_PATH_MONOMER = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/monomer_test/couplings/RASH_HUMAN_b03"
class TestCouplings(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCouplings, self).__init__(*args, **kwargs)
        self.raw_ec_file = "{}/couplings/test_new_ECs.txt".format(TRAVIS_PATH)
        self.couplings_file = "{}/couplings/test_new_CouplingScores.csv".format(TRAVIS_PATH)

        self.monomer_couplings = "{}_CouplingScores.csv".format(TRAVIS_PATH_MONOMER)
        self.enrichment_file = "{}_enrichment.csv".format(TRAVIS_PATH_MONOMER)

    def test_read_raw_ec_file(self):
        """
        tests the sorted and unsorted modes of read_raw_ec_file
        """
        ecs = pd.read_csv(self.raw_ec_file, sep=" ", names=["i", "A_i", "j", "A_j", "fn", "cn"])
        _test_ecs = read_raw_ec_file(self.raw_ec_file, sort=False, score="cn")

        pd.testing.assert_frame_equal(ecs, _test_ecs)

        sorted_ecs = ecs.sort_values(by="cn", ascending=False)
        _sorted_test_ecs = read_raw_ec_file(self.raw_ec_file, sort=True, score="cn")

        pd.testing.assert_frame_equal(sorted_ecs, _sorted_test_ecs)

    def test_enrichment(self):
        """
        tests the EC enrichment function
        """
        enrichment_scores = pd.read_csv(self.enrichment_file)
        ecs = pd.read_csv(self.monomer_couplings)
        _enrichment_scores = enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=6).reset_index(drop=True)
        pd.testing.assert_frame_equal(enrichment_scores, _enrichment_scores)

    def test_add_mixture_probability(self):
        pass
    def test_EVcomplexScoreModel_probability(self):
        """

        :return:
        """
        pass
if __name__ == '__main__':
    unittest.main()
