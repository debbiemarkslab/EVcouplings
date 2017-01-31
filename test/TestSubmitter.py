from unittest import TestCase
from evcouplings.utils import SubmitterFactory, Command, EResource

class TestSubmitter(TestCase):
    def setUp(self):
        pass

    # def test_submitter_factory_available(self):
    #     print(SubmitterFactory.available_methods())
    #
    # def test_submitter_factory_init(self):
    #     lsf = SubmitterFactory("lsf",blocking=True, db_path="test.db")
    #     assert lsf.isBlocking == True
    #
    # def minimal_LSF_example(self):
    #     lsf = SubmitterFactory("lsf")
    #     c = Command("sleep 1h",
    #                 name="test",
    #                 environment="source /home/bs224/.bashrc",
    #                 workdir="/home/bs224/",
    #                 resources={EResource.time: "01:01", EResource.queue: "short"})
    #
    #     job_id = lsf.submit(c)
    #     print(lsf.monitor(c))
    #     lsf.cancle(c)
    #     print(lsf.mointor(c))

    def test_local_submitter(self):
        print("Starting Test")
        local = SubmitterFactory("local", blocking=True, db_path="test.db")
        print(local)