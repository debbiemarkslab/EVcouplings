"""
Test cases for concatenation stage of EVCouplings complex pipeline

Author:
    Anna G. Green
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest import TestCase
from copy import deepcopy
import ruamel.yaml as yaml
from evcouplings.complex.alignment import *
from evcouplings.complex.distance import *
from evcouplings.complex.similarity import *
from evcouplings.complex.protocol import *
from evcouplings.align import Alignment

TRAVIS_PATH = os.getenv('HOME') + "/evcouplings_test_cases/complex_test"
# TRAVIS_PATH = "/home/travis/evcouplings_test_cases/complex_test"
# TRAVIS_PATH = "/Users/AG/Dropbox/evcouplings_dev/test_cases/for_B/complex_test"

class TestComplex(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestComplex, self).__init__(*args, **kwargs)

        # Genome location table
        genome_location_filename_1 = \
            "{}/align_1/test_new_genome_location.csv".format(TRAVIS_PATH)
        genome_location_filename_2 = \
            "{}/align_2/test_new_genome_location.csv".format(TRAVIS_PATH)

        self.gene_location_table_1 = pd.read_csv(genome_location_filename_1, header=0)
        self.gene_location_table_2 = pd.read_csv(genome_location_filename_2, header=0)

        # possible partners file
        possible_partners_file = "{}/concatenate/test_new_possible_partners.csv".format(TRAVIS_PATH)
        possible_partners = pd.read_csv(possible_partners_file, index_col=0, header=0,
            dtype={"uniprot_id_1": str, "uniprot_id_2": str, "distance": int}
        )
        possible_partners = possible_partners.sort_values(["uniprot_id_1", "uniprot_id_2", "distance"])
        self.possible_partners = possible_partners.reset_index(drop=True)

        # id pairing
        id_pairing_file = "{}/concatenate/test_new_id_pairing.csv".format(TRAVIS_PATH)
        id_pairing = pd.read_csv(id_pairing_file, index_col=0, header=0,
            dtype={"uniprot_id_1": str, "uniprot_id_2": str, "distance": int}
        )
        id_pairing = id_pairing.sort_values(["uniprot_id_1", "uniprot_id_2", "distance"])
        self.id_pairing = id_pairing.reset_index(drop=True)

        # annotation table for concatenation
        annotation_data_file = "{}/concatenate/test_new_uniprot_annotation.csv".format(TRAVIS_PATH)
        self.annotation_data = pd.read_csv(annotation_data_file, index_col=None, header=0, dtype=str)

        annotation_and_id_file = "{}/concatenate/test_new_annotation_and_id.csv".format(TRAVIS_PATH)
        self.annotation_and_id = pd.read_csv(
            annotation_and_id_file,  header=0, index_col=None,
            dtype={"id": str, "id_to_query": float, "species": str, "name": str}
        )

        # table of sequence identities
        similarities_file = "{}/align_1/test_new_identities.csv".format(TRAVIS_PATH)
        self.similarities = pd.read_csv(similarities_file)

        # table of paralogs
        paralog_file = "{}/concatenate/test_new_paralog_table.csv".format(TRAVIS_PATH)
        self.paralog_table = pd.read_csv(paralog_file, index_col=0, header=0)

        # input and output configuration
        with open("{}/concatenate/test_new_concatenate.incfg".format(TRAVIS_PATH)) as inf:
            self.incfg = yaml.safe_load(inf)

        with open("{}/concatenate/test_new_concatenate.outcfg".format(TRAVIS_PATH)) as inf:
            self.outcfg = yaml.safe_load(inf)

    def test_genome_distance(self):
        """
        tests the genome distance concatenation protocol.
        Verifies that all of the outfiles are created and non-empty (but does
        not check their contents).
        Verifies that the output configuration has all of the necessary keys
        """

        tmp_prefix = "tmp_"

        temporary_incfg = deepcopy(self.incfg)
        temporary_incfg["prefix"] = tmp_prefix
        temporary_incfg["first_alignment_file"] = "{}/align_1/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["second_alignment_file"] = "{}/align_2/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["first_genome_location_file"] = "{}/align_1/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["second_genome_location_file"] = "{}/align_2/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["first_annotation_file"] = "{}/align_1/test_new_annotation.csv".format(TRAVIS_PATH)
        temporary_incfg["second_annotation_file"] = "{}/align_2/test_new_annotation.csv".format(TRAVIS_PATH)

        outcfg = genome_distance(**temporary_incfg)

        # verify that the correct keys were created
        self.assertEqual(outcfg.keys(), self.outcfg.keys())

        # verify that all output files exist and are non-empty
        keys_list = [
            "raw_alignment_file",
            "first_concatenated_monomer_alignment_file",
            "second_concatenated_monomer_alignment_file",
            "statistics_file",
            "identities_file",
            "concatentation_statistics_file",
            "distance_plot_file",
            "alignment_file",
            "frequencies_file",
            "raw_focus_alignment_file"
        ]

        for key in keys_list:
            _file = outcfg[key]
            self.assertTrue(os.path.isfile(_file))
            self.assertTrue(os.path.getsize(_file) > 0)
            os.unlink(_file)

    def test_best_hit_normal(self):
        """
        tests the genome distance concatenation protocol.
        Verifies that all of the outfiles are created and non-empty (but does
        not check their contents).
        Verifies that the output configuration has all of the necessary keys
        """

        tmp_prefix = "tmp_"

        temporary_incfg = deepcopy(self.incfg)
        temporary_incfg["prefix"] = tmp_prefix
        temporary_incfg["first_alignment_file"] = "{}/align_1/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["second_alignment_file"] = "{}/align_2/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["first_annotation_file"] = "{}/align_1/test_new_annotation.csv".format(TRAVIS_PATH)
        temporary_incfg["second_annotation_file"] = "{}/align_2/test_new_annotation.csv".format(TRAVIS_PATH)
        temporary_incfg["first_identities_file"] = "{}/align_1/test_new_identities.csv".format(TRAVIS_PATH)
        temporary_incfg["second_identities_file"] = "{}/align_2/test_new_identities.csv".format(TRAVIS_PATH)
        temporary_incfg["first_genome_location_file"] = "{}/align_1/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["second_genome_location_file"] = "{}/align_2/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["use_best_reciprocal"] = False
        temporary_incfg["paralog_identity_threshold"] = 0.9

        with open("{}/concatenate/test_new_best_hit_concatenate.outcfg".format(TRAVIS_PATH)) as inf:
            _outcfg = yaml.safe_load(inf)

        outcfg = best_hit(**temporary_incfg)

        # verify that the correct keys were created
        self.assertEqual(outcfg.keys(), _outcfg.keys())

        # verify that all output files exist and are non-empty
        keys_list = [
            "raw_alignment_file",
            "first_concatenated_monomer_alignment_file",
            "second_concatenated_monomer_alignment_file",
            "statistics_file",
            "identities_file",
            "concatentation_statistics_file",
            "alignment_file",
            "frequencies_file",
            "raw_focus_alignment_file"
        ]

        for key in keys_list:
            _file = outcfg[key]
            self.assertTrue(os.path.isfile(_file))
            self.assertTrue(os.path.getsize(_file) > 0)
            os.unlink(_file)

    def test_best_hit_reciprocal(self):
        """
        tests the genome distance concatenation protocol.
        Verifies that all of the outfiles are created and non-empty (but does
        not check their contents).
        Verifies that the output configuration has all of the necessary keys
        """

        tmp_prefix = "tmp_"

        temporary_incfg = deepcopy(self.incfg)
        temporary_incfg["prefix"] = tmp_prefix
        temporary_incfg["first_alignment_file"] = "{}/align_1/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["second_alignment_file"] = "{}/align_2/test_new.a2m".format(TRAVIS_PATH)
        temporary_incfg["first_annotation_file"] = "{}/align_1/test_new_annotation.csv".format(TRAVIS_PATH)
        temporary_incfg["second_annotation_file"] = "{}/align_2/test_new_annotation.csv".format(TRAVIS_PATH)
        temporary_incfg["first_identities_file"] = "{}/align_1/test_new_identities.csv".format(TRAVIS_PATH)
        temporary_incfg["second_identities_file"] = "{}/align_2/test_new_identities.csv".format(TRAVIS_PATH)
        temporary_incfg["first_genome_location_file"] = "{}/align_1/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["second_genome_location_file"] = "{}/align_2/test_new_genome_location.csv".format(TRAVIS_PATH)
        temporary_incfg["use_best_reciprocal"] = True
        temporary_incfg["paralog_identity_threshold"] = 0.9

        with open("{}/concatenate/test_new_best_reciprocal_concatenate.outcfg".format(TRAVIS_PATH)) as inf:
            _outcfg = yaml.safe_load(inf)

        outcfg = best_hit(**temporary_incfg)

        # verify that the correct keys were created
        self.assertEqual(outcfg.keys(), _outcfg.keys())

        # verify that all output files exist and are non-empty
        keys_list = [
            "raw_alignment_file",
            "first_concatenated_monomer_alignment_file",
            "second_concatenated_monomer_alignment_file",
            "statistics_file",
            "identities_file",
            "concatentation_statistics_file",
            "alignment_file",
            "frequencies_file",
            "raw_focus_alignment_file"
        ]

        for key in keys_list:
            _file = outcfg[key]
            self.assertTrue(os.path.isfile(_file))
            self.assertTrue(os.path.getsize(_file) > 0)
            os.unlink(_file)

    def test_modify_complex_segments(self):
        """
        tests that modify_complex_segments adds the correct "segments" field to the outcfg dictionary
        :return:
        """
        test_configuration = deepcopy(self.incfg)
        cfg = deepcopy(test_configuration)
        # in pipeline, cfg is the output configuration
        # need to provide dummy here since our 'gold standard' outcfg
        # already has segments field
        _outcfg = modify_complex_segments(cfg, **test_configuration)

        test_configuration["segments"] = self.outcfg["segments"]
        self.assertDictEqual(test_configuration, _outcfg)

    def test_describe_concatenation(self):
        """
        test whether describe_concatenation writes a file with the correct contents
        """
        outfile = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)

        describe_concatenation(
            "{}/align_1/test_new_annotation.csv".format(TRAVIS_PATH),
            "{}/align_2/test_new_annotation.csv".format(TRAVIS_PATH),
            "{}/align_1/test_new_genome_location.csv".format(TRAVIS_PATH),
            "{}/align_2/test_new_genome_location.csv".format(TRAVIS_PATH),
            outfile.name
        )

        concatenation_stats = pd.read_csv("{}/concatenate/test_new_concatenation_statistics.csv".format(
            TRAVIS_PATH
        ))
        _concatenation_stats = pd.read_csv(outfile.name)

        pd.testing.assert_frame_equal(concatenation_stats, _concatenation_stats)
        os.unlink(outfile.name)

    def test_write_concatenated_alignment(self):
        """
        tests whether write_concatenated alignment returns the correct 3
        alignment objects

        """
        alignment_1_file = "{}/concatenate/test_new_monomer_1.fasta".format(TRAVIS_PATH)
        with(open(alignment_1_file)) as inf:
            alignment_1 = Alignment.from_file(inf)

        alignment_2_file = "{}/concatenate/test_new_monomer_2.fasta".format(TRAVIS_PATH)
        with(open(alignment_2_file)) as inf:
            alignment_2 = Alignment.from_file(inf)

        alignment_file = "{}/concatenate/test_new_raw_focus.fasta".format(TRAVIS_PATH)
        with(open(alignment_file)) as inf:
            alignment = Alignment.from_file(inf)

        target_header = alignment.ids[0]

        input_alignment_file_1 = "{}/align_1/test_new.a2m".format(TRAVIS_PATH)
        input_alignment_file_2 = "{}/align_2/test_new.a2m".format(TRAVIS_PATH)

        id_pairing = deepcopy(self.id_pairing)
        id_pairing.loc[:, "id_1"] = id_pairing["uniprot_id_1"]
        id_pairing.loc[:, "id_2"] = id_pairing["uniprot_id_2"]

        _target_header, _target_seq_idx, _ali, _ali_1, _ali_2 = write_concatenated_alignment(
            id_pairing, input_alignment_file_1, input_alignment_file_2,
            "DINJ_ECOLI/1-86", "YAFQ_ECOLI/1-92"
        )

        def _test_aln_equivalence(ali1, ali2):
            np.testing.assert_array_equal(ali1.ids, ali2.ids)
            np.testing.assert_array_equal(ali1.matrix, ali2.matrix)
#
        _test_aln_equivalence(alignment_1, _ali_1)
        _test_aln_equivalence(alignment_2, _ali_2)
        _test_aln_equivalence(alignment, _ali)
        self.assertEqual(target_header, _target_header)
        self.assertEqual(_target_seq_idx, 0)

    def test_read_species_annotation_table_uniprot(self):
        """
        tests whether a uniprot annotation table is read correctly

        """
        annotation_file_uniprot = "{}/align_1/test_new_annotation.csv".format(TRAVIS_PATH)
        annotation_data = read_species_annotation_table(annotation_file_uniprot)
        pd.testing.assert_frame_equal(annotation_data, self.annotation_data)

    def test_read_species_annotation_table_uniref(self):
        """
        tests whether a uniref annotation table is read correctly

        """
        _annotation_file_uniref = "{}/DIVIB_BACSU_1-54_b0.3_annotation.csv".format(TRAVIS_PATH)
        _annotation_data = read_species_annotation_table(_annotation_file_uniref)
        annotation_file_uniref = "{}/concatenate/test_new_uniref_annotation.csv".format(TRAVIS_PATH)
        annotation_data_gold = pd.read_csv(annotation_file_uniref, index_col=None, header=0, dtype=str)
        pd.testing.assert_frame_equal(annotation_data_gold, _annotation_data)

    def test_most_similar_by_organism(self):
        """
        tests whether most_similar_by_organism returns the correct dataframe

        """
        annotation_and_id = most_similar_by_organism(self.similarities, self.annotation_data)
        pd.testing.assert_frame_equal(annotation_and_id, self.annotation_and_id)

    def test_find_paralogs(self):
        """
        tests whether find_paralogs returns the correct dataframe

        """
        target_id = "DINJ_ECOLI"
        paralog_table = find_paralogs(target_id, self.annotation_data, self.similarities, 0.9)
        pd.testing.assert_frame_equal(paralog_table, self.paralog_table)

    def test_filter_best_reciprocal(self):
        """
        tests whether filter_best_reciprocal returns the correct dataframe

        """
        alignment_file = "{}/align_1/test_new.a2m".format(TRAVIS_PATH)
        best_recip = pd.read_csv("{}/concatenate/test_new_best_reciprocal.csv".format(TRAVIS_PATH), index_col=0)
        _best_recip = filter_best_reciprocal(alignment_file, self.paralog_table, self.annotation_and_id, 0.02)
        pd.testing.assert_frame_equal(best_recip, _best_recip)

    def test_get_distance_overlap(self):
        """
        tests whether get_distance returns 0 for overlapping genes

        """
        annotation_1 = (1000, 1500)
        annotation_2 = (1400, 1700)
        distance = get_distance(annotation_1, annotation_2)
        self.assertEqual(distance, 0)

    def test_get_distance_reverse(self):
        """
        tests whether get distance correctly measures distance of two genes with opposite strand

        """
        annotation_1 = (1000, 1500)
        annotation_2 = (1800, 1700)
        distance = get_distance(annotation_1, annotation_2)
        self.assertEqual(distance, 200)

    def test_get_distance_increasing(self):
        """
        tests whether get_distance correctly measures distance of two genes with same strand

        """
        annotation_1 = (1000, 1500)
        annotation_2 = (1700, 1800)
        distance = get_distance(annotation_1, annotation_2)
        self.assertEqual(distance, 200)

    def test_best_reciprocal_matching(self):
        """
        tests whether best_reciprocal_matchin generates the correct pd.DataFrame

        """
        id_pairing = best_reciprocal_matching(self.possible_partners)
        id_pairing = id_pairing.sort_values(["uniprot_id_1", "uniprot_id_2", "distance"])
        id_pairing = id_pairing.reset_index(drop=True)
        pd.testing.assert_frame_equal(id_pairing, self.id_pairing)

    def test_find_possible_partners(self):
        """
        tests whether find_possible partners generates the correct pd.DataFrame

        """

        _possible_partners = find_possible_partners(
            self.gene_location_table_1,
            self.gene_location_table_2
        )
        _possible_partners = _possible_partners.sort_values(["uniprot_id_1", "uniprot_id_2", "distance"])
        _possible_partners = _possible_partners.reset_index(drop=True)

        pd.testing.assert_frame_equal(
            self.possible_partners, _possible_partners,
            check_less_precise=True, check_like=True,
            check_names=False
        )

    def test_plot_distance_distribution(self):
        """
        tests whether plot_distance_distribution generates a non-empty PDF

        """
        outfile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        plot_distance_distribution(self.id_pairing, outfile.name)
        self.assertTrue(os.path.isfile(outfile.name))
        self.assertTrue(os.path.getsize(outfile.name) > 0)
        outfile.close()
        os.unlink(outfile.name)

if __name__ == '__main__':
    unittest.main()
