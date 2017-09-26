import os
import unittest
from unittest import TestCase
from evcouplings.utils import SubmitterFactory, Command
from evcouplings.utils.app import app


class TestSubmitter(TestCase):

    def test_submitter_factory_available(self):
        print(SubmitterFactory.available_methods())

    def test_submitter_factory_init(self):
        lsf = SubmitterFactory("lsf", blocking=True, db_path="test.db")
        assert lsf.isBlocking == True

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
        local = SubmitterFactory("local", blocking=True, db_path="./test/test.db")
        c = Command("sleep 5 && touch ./test/test_submission.txt", name="test_sleep")

        local.submit(c)
        assert local.monitor(c) in ["done", "susp", "run", "pend"]
        local.join()
        assert local.monitor(c) in ["done", "exit"]
        assert os.path.exists("./test/test_submission.txt")
        os.remove("./test/test_submission.txt")
        os.remove("./test/test.db")

    def test_local_cancel(self):
        local = SubmitterFactory("local", blocking=True, db_path="./test/test.db")
        c = Command("sleep 5 && touch ./test/test_submission.txt", name="test_sleep")

        local.submit(c)
        assert local.cancel(c)
        os.remove("./test/test.db")

    def test_local_dependency(self):
        local = SubmitterFactory("local", blocking=True, db_path="./test/test.db")
        c1 = Command("sleep 5 ", name="test_sleep")
        c2 = Command("touch ./test/test_submission.txt", name="test_touch")

        local.submit(c1)
        local.submit(c2, dependent=c1)
        x = local.monitor(c2)

        assert  x in ["susp", "pend"]
        local.join()
        assert os.path.exists("./test/test_submission.txt")
        os.remove("./test/test_submission.txt")
        os.remove("./test/test.db")


if __name__ == '__main__':
    unittest.main()