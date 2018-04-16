import os
import unittest
from unittest import TestCase
from evcouplings.utils import SubmitterFactory, Command, ResourceError, tempfile
from evcouplings.utils.app import *


class TestUtilsAppp(TestCase):

    def setUp(self):
        self.config_file = b"""
stages:
    - align
global:
    prefix: test
    sequence_id:
    sequence_file:
    region:
    theta: 0.8
    cpu:
align:
    protocol: standard
environment:
    engine: slurm
    queue: medium
    cores: 2
databases:
    uniprot: /n/groups/marks/databases/jackhmmer/uniprot/uniprot_current.fasta    
"""

    def test_substitute_config_ResourceError(self):
        """
        tests whether the ResourceError

        """
        kwargs = {"config":None}
        self.assertRaises(ResourceError, substitute_config, **kwargs)

    def test_substitute_config_cores(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "cores": "4"}
        res = substitute_config(**kwargs)
        self.assertEqual("4", res["global"]["cpu"])
        os.unlink(tmp.name)

    def test_substitute_config_alignment(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "alignment": "test.fasta"}
        res = substitute_config(**kwargs)
        self.assertEqual("existing", res["align"]["protocol"])
        os.unlink(tmp.name)

    def test_substitute_config_region_InvalidParameterError(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "region": "123.1231"}
        self.assertRaises(InvalidParameterError, substitute_config, **kwargs)
        os.unlink(tmp.name)

    def test_substitute_config_region(self):
        """
        Test whether a region is correctly substituted
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "region": "1-10"}
        res = substitute_config(**kwargs)
        self.assertEqual([1, 10], res["global"]["region"])
        os.unlink(tmp.name)

    def test_substitute_config_stages(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "stages": "test"}
        res = substitute_config(**kwargs)
        self.assertEqual(["test"], res["stages"])
        os.unlink(tmp.name)

    def test_substitute_config_existing_db(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "database": "uniprot"}
        res = substitute_config(**kwargs)
        self.assertEqual("uniprot", res["align"]["database"])
        os.unlink(tmp.name)

    def test_substitute_config_custom_db(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "database": "uniref"}
        res = substitute_config(**kwargs)
        self.assertEqual("custom", res["align"]["database"])
        os.unlink(tmp.name)

    def test_substitute_config_biscore_error(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "bitscores": "1.0", "evalues": "1.0"}
        self.assertRaises(InvalidParameterError, substitute_config, **kwargs)
        os.unlink(tmp.name)

    def test_substitute_config_single_bitscore(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "bitscores": "1.0"}
        res = substitute_config(**kwargs)
        self.assertEqual(1.0, res["align"]["use_bitscores"])
        self.assertEqual(1.0, res["align"]["domain_threshold"])
        os.unlink(tmp.name)

    def test_substitute_config_bitscore_not_numerical(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "bitscores": "one"}
        self.assertRaises(InvalidParameterError, substitute_config, **kwargs)
        os.unlink(tmp.name)

    def test_substitute_config_multiple_bitscore(self):
        """
        Test whether a InvalidPrama Error is correctly raised
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "bitscores": "1.0,0.7"}
        res = substitute_config(**kwargs)
        self.assertTrue("batch" in res)
        os.unlink(tmp.name)

    def test_unroll_config_no_batch(self):
        """
        test whether config is not unrolled if batch is not defined
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name}
        res = substitute_config(**kwargs)
        res = unroll_config(res)
        self.assertTrue("test" in res)
        os.unlink(tmp.name)

    def test_unroll_config_batch(self):
        """
        test whether config is not unrolled if batch is not defined
        """
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(self.config_file)
        tmp.close()
        kwargs = {"config": tmp.name, "bitscores": "1.0,0.7"}
        res = substitute_config(**kwargs)
        res = unroll_config(res)
        self.assertTrue(len(res) == 2)
        os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()


