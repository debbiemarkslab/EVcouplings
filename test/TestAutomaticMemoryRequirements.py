import pkg_resources
import unittest
from unittest import TestCase
from copy import deepcopy
from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import calculate_memory_requirements


class TestAutomaticMemoryRequirements(TestCase):
    def setUp(self):
        # Test files have to be included into the python package during installation
        config_file_mono = "../config/sample_config_monomer.txt"
        self.config_mono = read_config_file(config_file_mono, preserve_order=True)

        config_file_complex = "../config/sample_config_complex.txt"
        self.config_complex = read_config_file(config_file_complex, preserve_order=True)

    def test_mono_range(self):
        """
        Test the range calculation
        """
        cfg = deepcopy(self.config_mono)
        cfg["global"]["region"] = (0, 10)
        cfg["couplings"]["ignore_gaps"] = False

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 1.6044)

    def test_mono_custom_alphabet(self):
        """
        tests the custom alphabet calculation
        and gap-ignore function
        """

        cfg = deepcopy(self.config_mono)
        cfg["global"]["region"] = (0, 10)
        cfg["couplings"]["alphabet"] = "-ABC"
        cfg["couplings"]["ignore_gaps"] = False

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 0.0608)

        # gap-ignore enabled
        cfg["couplings"]["ignore_gaps"] = True
        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 0.0348)

    def test_mono_id(self):
        """
        tests automatic sequence retrieval memory calculation
        """

        cfg = deepcopy(self.config_mono)
        cfg["global"]["sequence_id"] = "A0A1L5JMI0_PROVU"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 449.1648)

    def test_mono_fasta(self):
        """
        tests fasta-based  memory calculation
        """

        cfg = deepcopy(self.config_mono)
        cfg["global"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8_raw_focus.fasta"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 125.4544)

    def test_mono_stockholm(self):
        """
        tests stockholm-based memory calculation
        """

        cfg = deepcopy(self.config_mono)
        cfg["global"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8.sto"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 454.5424)

    def test_complex_range(self):
        """
        Test the range calculation
        """
        cfg = deepcopy(self.config_complex)
        cfg["align_1"]["region"] = (0, 5)
        cfg["align_2"]["region"] = (0, 5)
        cfg["couplings"]["ignore_gaps"] = False

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 1.6044)

    def test_complex_custom_alphabet(self):
        """
        tests the custom alphabet calculation
        and gap-ignore function
        """

        cfg = deepcopy(self.config_complex)
        cfg["align_1"]["region"] = (0, 5)
        cfg["align_2"]["region"] = (0, 5)
        cfg["couplings"]["alphabet"] = "-ABC"
        cfg["couplings"]["ignore_gaps"] = False

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 0.0608)

        # gap-ignore enabled
        cfg["couplings"]["ignore_gaps"] = True
        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 0.0348)

    def test_complex_id(self):
        """
        tests automatic sequence retrieval memory calculation
        """

        cfg = deepcopy(self.config_complex)
        cfg["align_1"]["sequence_id"] = "A0A1L5JMI0_PROVU"
        cfg["align_2"]["sequence_id"] = "A0A1L5JMI0_PROVU"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 1801.4976)

    def test_complex_fasta(self):
        """
        tests fasta-based  memory calculation
        """

        cfg = deepcopy(self.config_complex)
        cfg["align_1"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8_raw_focus.fasta"
        cfg["align_2"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8_raw_focus.fasta"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 504.3808)

    def test_complex_stockholm(self):
        """
        tests stockholm-based memory calculation
        """

        cfg = deepcopy(self.config_complex)
        cfg["align_1"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8.sto"
        cfg["align_2"]["sequence_file"] = "data/RS12_ECOLI_1-89/align/RS15_ECOLI_1-89_b0.8.sto"

        calc_mem = calculate_memory_requirements(cfg)
        self.assertAlmostEqual(calc_mem, 1823.0368)

if __name__ == '__main__':
    unittest.main()