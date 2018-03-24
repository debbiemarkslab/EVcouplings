"""
Test cases for mutate stage of EVCouplings pipline

Author:
    Anna G. Green
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest import TestCase
from evcouplings.mutate.calculations import *
from evcouplings.mutate.protocol import *
from evcouplings.couplings.model import CouplingsModel

TRAVIS_PATH = "/home/travis/evcouplings_test_cases/monomer_test"
# TRAVIS_PATH = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/monomer_test"


class TestMutation(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMutation, self).__init__(*args, **kwargs)

        self.model_file = "{}/couplings/RASH_HUMAN_b03.model".format(TRAVIS_PATH)
        self.c = CouplingsModel(self.model_file)
        self.c0 = self.c.to_independent_model()

        self.singles = pd.read_csv(
            "{}/mutate/RASH_HUMAN_b03_single_mutant_matrix.csv".format(TRAVIS_PATH)
        )

    def test_extract_mutations_normal(self):
        """
        tests whether extract mutations correctly separates mutation strings
        :return:
        """
        mutation_string = "A143K,M100K"
        mutation_target = [
            (143, "A", "K"),
            (100, "M", "K")
        ]
        mutations = extract_mutations(mutation_string)
        self.assertEqual(mutations, mutation_target)

    def test_extract_mutations_null_input(self):
        """
        test whether extract mutations returns nothing if given an incorrect string
        :return:
        """
        mutation_string = ""
        mutation_target = []
        mutations = extract_mutations(mutation_string)
        self.assertEqual(mutations, mutation_target)

    def test_extract_mutations_wt(self):
        """
        test whether extract mutations returns wt value if given "wt" string
        :return:
        """
        mutation_string = "wt"
        mutation_target = []
        mutations = extract_mutations(mutation_string)
        self.assertEqual(mutations, mutation_target)

    def test_predict_mutation_table(self):
        """
        tests whether predict mutation table returns a correct table of mutations
        :return:
        """
        singles = self.singles.drop("prediction_independent", axis=1)

        _singles = predict_mutation_table(
            self.c0, singles, output_column="prediction_independent"
        )

        pd.testing.assert_frame_equal(self.singles, _singles)

    def test_single_mutant_matrix(self):
        """
        tests whether single mutant matrix returns the correct pd.DataFrame
        :return:
        """
        _singles = single_mutant_matrix(
            self.c, output_column="prediction_epistatic"
        )

        singles = self.singles.drop("prediction_independent", axis=1)

        pd.testing.assert_frame_equal(singles, _singles)

    def test_split_mutants_single(self):
        """

        :return:
        """
        mutations = [
            "A124K",
            "M122L",
            "A156V"
        ]

        data = pd.DataFrame({
            "mutations": mutations
        })

        output = pd.DataFrame({
            "pos": [124, 122, 156],
            "wt": ["A", "M", "A"],
            "subs": ["K", "L", "V"],
            "mutations": mutations,
            "num_mutations": [1, 1, 1]
        }, dtype=object)

        output = output[[
            "mutations", "num_mutations", "pos", "wt", "subs"
        ]]
        output["num_mutations"] = output["num_mutations"].astype(int)
        output["pos"] = output["pos"].astype(str)

        split_mutations = split_mutants(data, "mutations")

        pd.testing.assert_frame_equal(output, split_mutations)

    def test_split_mutants_double(self):
        """

        :return:
        """
        mutations = [
            "A124K,W145Y",
            "M122L",
            "A156V"
        ]

        data = pd.DataFrame({
            "mutations": mutations
        })

        output = pd.DataFrame({
            "pos": ["124,145", "122", "156"],
            "wt": ["A,W", "M", "A"],
            "subs": ["K,Y", "L", "V"],
            "mutations": mutations,
            "num_mutations": [2, 1, 1]
        }, dtype=object)

        output = output[[
            "mutations", "num_mutations", "pos", "wt", "subs"
        ]]
        output["num_mutations"] = output["num_mutations"].astype(int)
        output["pos"] = output["pos"].astype(str)

        split_mutations = split_mutants(data, "mutations")

        pd.testing.assert_frame_equal(output, split_mutations)

# def test_protcol_standard(self):
    #     """
    #     TODO: fix circular dependency problem. Currently, lines 80, 85, and 106 of the mutate protocol will throw an error 
    #     in testing because of their import statements. DO NOT change the import statements in the protocol.py itself else
    #     you will break the protocol in production (even if it works in testing) 
    #     """
    #     tmp_prefix = "tmp_"

    #     outcfg = standard(**{
    #         "prefix": tmp_prefix,
    #         "model_file": self.model_file,
    #         "mutation_dataset_file": "{}/mutate/RASH_HUMAN_b03_mutation_dataset.csv".format(TRAVIS_PATH)
    #     })
    #     file_output_keys = [
    #         "mutation_matrix_plot_files",
    #         "mutation_matrix_file",
    #         "mutations_epistatic_pml_files",
    #         "mutation_dataset_predicted_file"
    #     ]

    #     for key in file_output_keys:
    #         print(key)
    #         if type(outcfg[key]) is str:
    #             _file = outcfg[key]
    #             self.assertTrue(os.path.isfile(_file))
    #             self.assertTrue(os.path.getsize(_file) > 0)
    #             os.unlink(_file)
    #         else:
    #             for _file in outcfg[key]:
    #                 self.assertTrue(os.path.isfile(_file))
    #                 self.assertTrue(os.path.getsize(_file) > 0)
    #                 os.unlink(_file)


if __name__ == '__main__':
    unittest.main()
