"""
Test cases for couplings stage of EVCouplingspipeline

Author:
    Anna G. Green
"""

import unittest
import os
import tempfile
import pandas as pd
import ruamel.yaml as yaml
from unittest import TestCase
from evcouplings.couplings.pairs import *

TRAVIS_PATH = "/home/travis/evcouplings_test_cases/complex_test"
TRAVIS_PATH_MONOMER = "/home/travis/evcouplings_test_cases/monomer_test/couplings/RASH_HUMAN_b03"

#TRAVIS_PATH = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/complex_test"
#TRAVIS_PATH_MONOMER = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/monomer_test/couplings/RASH_HUMAN_b03"
class TestCouplings(TestCase):


    def __init__(self, *args, **kwargs):
        super(TestCouplings, self).__init__(*args, **kwargs)
        self.raw_ec_file = "{}/couplings/test_new_ECs.txt".format(TRAVIS_PATH)
        self.couplings_file = "{}/couplings/test_new_CouplingScores.csv".format(TRAVIS_PATH)
        self.ecs = pd.read_csv(self.couplings_file)
        self.inter_ecs = self.ecs.query("segment_i != segment_j")

        with open("{}/couplings/test_new_couplings.outcfg".format(TRAVIS_PATH)) as inf:
            self.outcfg = yaml.safe_load(inf)

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
        tests the EC enrichment function when the monomer ECs have no segments
        """
        enrichment_scores = pd.read_csv(self.enrichment_file)
        ecs = pd.read_csv(self.monomer_couplings)
        ecs = ecs.drop(["segment_i", "segment_j"], axis=1)
        _enrichment_scores = enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=6).reset_index(drop=True)

        pd.testing.assert_frame_equal(enrichment_scores, _enrichment_scores)

    def test_enrichment_monomer_segment(self):
        """
        tests the EC enrichment function when the monomer ECs have segments
        """
        enrichment_scores = pd.read_csv(self.enrichment_file)
        enrichment_scores["segment_i"] = "A"
        enrichment_scores = enrichment_scores[['i', 'A_i', 'segment_i', 'enrichment']]

        ecs = pd.read_csv(self.monomer_couplings)
        _enrichment_scores = enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=6).reset_index(drop=True)

        pd.testing.assert_frame_equal(enrichment_scores, _enrichment_scores)

    def test_enrichment_complex_segment(self):
        """
        tests the EC enrichment function with complex ECs
        """

        ecs = pd.read_csv(self.couplings_file)
        _enrichment_scores = enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=6).reset_index(drop=True)
        enrichment_scores = pd.read_csv(("{0}/couplings/test_new_enrichment.csv".format(TRAVIS_PATH)))
        pd.testing.assert_frame_equal(enrichment_scores, _enrichment_scores)

    def test_add_mixture_probability_skewnormal(self):
        ecs_with_prob = add_mixture_probability(
            self.inter_ecs, model="skewnormal"
        )

        _test_ecs = pd.read_csv(
            "{}/couplings/test_new_CouplingScores_skewnormal.csv".format(TRAVIS_PATH), index_col=0
        )
        pd.testing.assert_frame_equal(ecs_with_prob, _test_ecs)

    def test_add_mixture_probability_normal(self):
        ecs_with_prob = add_mixture_probability(
            self.inter_ecs, model="normal"
        )

        _test_ecs = pd.read_csv(
            "{}/couplings/test_new_CouplingScores_normal.csv".format(TRAVIS_PATH), index_col=0
        )
        pd.testing.assert_frame_equal(ecs_with_prob, _test_ecs)

    def test_add_mixture_probability_evcomplex_uncorrected(self):
        ecs_with_prob = add_mixture_probability(
            self.inter_ecs, model="evcomplex_uncorrected"
        )

        _test_ecs = pd.read_csv(
            "{}/couplings/test_new_CouplingScores_evc_raw.csv".format(TRAVIS_PATH), index_col=0
        )
        pd.testing.assert_frame_equal(ecs_with_prob, _test_ecs)

    def test_add_mixture_probability_evcomplex(self):
        NeffL = self.outcfg["effective_sequences"] / self.outcfg["num_sites"]
        ecs_with_prob = add_mixture_probability(
            self.inter_ecs, model="evcomplex", N_effL=NeffL
        )

        _test_ecs = pd.read_csv(
            "{}/couplings/test_new_CouplingScores_evc.csv".format(TRAVIS_PATH), index_col=0
        )
        pd.testing.assert_frame_equal(ecs_with_prob, _test_ecs)

    def test_add_mixture_probability_evcomplex_error(self):
        """
        tests that EVcomplex score cannot be calculated without a user-supplied Meff
        :return:
        """
        with self.assertRaises(ValueError):
            add_mixture_probability(self.ecs, model="evcomplex")

    def test_add_mixture_probability_invalid_selection(self):
        with self.assertRaises(ValueError):
            add_mixture_probability(self.ecs, model="fake news")

if __name__ == '__main__':
    unittest.main()
