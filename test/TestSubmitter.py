from unittest import TestCase
from evcouplings.utils import SubmitterFactory


class TestSubmitter(TestCase):
    def setUp(self):
        pass

    def test_submitter_factory_available(self):
        print(SubmitterFactory.available_methods())

    def test_submitter_factory_init(self):
        lsf = SubmitterFactory("lsf",blocking=True,db_path="test.db")
        assert lsf.isBlocking == True