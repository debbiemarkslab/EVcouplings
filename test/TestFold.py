"""
Test cases for fold stage of EVCouplings pipline
Currently only tests complex_dock protocol and its called functions

TODO: fix circular import of COMPLEX_CHAIN_NAMES

Author:
    Anna G. Green
"""

import unittest
import os
import ruamel.yaml as yaml
from unittest import TestCase
from evcouplings.fold import haddock_dist_restraint
from evcouplings.fold.protocol import complex_dock

TRAVIS_PATH = os.getenv('HOME') + "/evcouplings_test_cases"
# TRAVIS_PATH = "/home/travis/evcouplings_test_cases"
#TRAVIS_PATH = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B"
COMPLEX_PATH = "{}/complex_test".format(TRAVIS_PATH)

class TestComplexDock(TestCase):
    """
    NOTE: not explicitly testing evcouplings.fold.restraints.docking_restraints
        because there is no output of this function - this function is called
        by the protocol, and only calls haddock_dist_restraint and writes the
        output to a file
    """

    def __init__(self, *args, **kwargs):
        super(TestComplexDock, self).__init__(*args, **kwargs)

        config_file = COMPLEX_PATH + "/couplings/test_new_couplings.outcfg"
        config = yaml.safe_load(open(config_file, "r"))

        self.ec_file = COMPLEX_PATH + "/couplings/test_new_CouplingScores.csv"
        self.segments = config["segments"]

    def test_haddock_restraint_no_comment(self):
        """
        test whether evcouplings.fold.haddock.haddock_dist_restraint returns
        the correct string
        :return:
        """
        r = haddock_dist_restraint(
            "10", "A", "11", "B",
            "1.0", "0.0", "2.0",
        )

        desired_output = (
            "! \n"
            "assign (resid 10 and segid A)\n"
            "(\n"
            " (resid 11 and segid B)\n"
            ") 1.0 2.0 0.0"
        )

        self.assertEqual(r, desired_output)

    def test_haddock_restraint_with_comment(self):
        """
        test whether evcouplings.fold.haddock.haddock_dist_restraint returns
        the correct string
        :return:
        """
        r = haddock_dist_restraint(
            "10", "A", "11", "B",
            "1.0", "0.0", "2.0", comment = "COMMENT"
        )

        desired_output = (
            "! COMMENT\n"
            "assign (resid 10 and segid A)\n"
            "(\n"
            " (resid 11 and segid B)\n"
            ") 1.0 2.0 0.0"
        )

        self.assertEqual(r, desired_output)

    # def test_haddock_restraint_with_atom(self):
    #     """
    #     test whether evcouplings.fold.haddock.haddock_dist_restraint returns
    #     the correct string
    #     :return:
    #     """
    #     r = haddock_dist_restraint(
    #         "10", "A", "11", "B",
    #         "1.0", "0.0", "2.0", atom_i = "CA",
    #         atom_j="CA", comment="COMMENT"
    #     )
    #
    #     desired_output = (
    #         "! COMMENT\n"
    #         "assign (resid 10 and segid A and name CA)\n"
    #         "(\n"
    #         " (resid 11 and segid B and name CA)\n"
    #         ") 1.0 2.0 0.0"
    #     )
    #
    #     self.assertEqual(r, desired_output)


    def test_protocol(self):
        """
        test whether evcouplings.fold.protocol.complex_dock writes the correct
        files

        :return:
        """

        tmp_prefix = "tmp_"

        outcfg = complex_dock(
            prefix = tmp_prefix,
            ec_file = self.ec_file,
            segments = self.segments,
            dock_probability_cutoffs = [0.9, 0.99],
            dock_lowest_count = 5,
            dock_highest_count = 10,
            dock_increase = 5
        )

        file_output_keys = [
            "tmp__significant_ECs_0.9_restraints.tbl",
            "tmp__significant_ECs_0.99_restraints.tbl",
            "tmp__5_restraints.tbl",
            "tmp__10_restraints.tbl"
        ]

        for _file in file_output_keys:
            self.assertTrue(os.path.isfile(_file))
            self.assertTrue(os.path.getsize(_file) > 0)
            os.unlink(_file)

if __name__ == '__main__':
    unittest.main()
