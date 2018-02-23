import os
import unittest
from unittest import TestCase
import evcouplings
from evcouplings.utils import SubmitterFactory, Command


class TestSubmitter(TestCase):

    def setUp(self):

        self.module_path = os.path.dirname(evcouplings.__file__)
        self.test_db = os.path.join("./", "test", "test.db")
        self.test_submission = os.path.join("./", "test", "test_submission.txt")

    def test_submitter_factory_available(self):
        print(SubmitterFactory.available_methods())

    def test_submitter_factory_init(self):
        lsf = SubmitterFactory("lsf", blocking=True, db_path="test.db")
        self.assertTrue(lsf.isBlocking)

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
        local = SubmitterFactory("local", blocking=True, db_path=self.test_db)
        c = Command("sleep 1 && touch {}".format(self.test_submission), name="test_sleep")

        local.submit(c)
        self.assertTrue(local.monitor(c) in ["done", "susp", "run", "pend"])
        local.join()
        self.assertTrue(local.monitor(c) in ["done", "exit"])
        self.assertTrue(os.path.exists(self.test_submission))
        os.remove(self.test_submission)
        os.remove(self.test_db)

    def test_local_cancel(self):
        local = SubmitterFactory("local", blocking=True, db_path=self.test_db)
        c = Command("sleep 1 && touch {}".format(self.test_submission), name="test_sleep")

        local.submit(c)
        self.assertTrue(local.cancel(c))
        os.remove(self.test_db)

    def test_local_dependency(self):
        local = SubmitterFactory("local", blocking=True, db_path=self.test_db)
        c1 = Command("sleep 1 ", name="test_sleep")
        c2 = Command("sleep 1 && touch {}".format(self.test_submission), name="test_touch")

        local.submit(c1)
        local.submit(c2, dependent=c1)
        x = local.monitor(c2)

        self.assertTrue(x in ["susp", "pend"])
        local.join()
        self.assertTrue(os.path.exists(self.test_submission))
        os.remove(self.test_submission)
        os.remove(self.test_db)


if __name__ == '__main__':
    unittest.main()
