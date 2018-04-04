import os
import unittest
from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evcouplings.utils.database import update_job_status, EStatus, ComputeJob


class TestUtilsDatabase(TestCase):

    def setUp(self):
        self.config = {
            "global": {
                "prefix": "test"
                      },
            "management": {
                 "database_uri": 'sqlite:///test.db',
                 "job_name": "test_case"
                           }
        }

    def test_update_job_status(self):
        """
        test whether job status is changed
        """
        mgmt = self.config.get("management", {})
        uri = mgmt.get("database_uri", None)
        job_name = mgmt.get("job_name", None)
        update_job_status(self.config, status=EStatus.RUN)

        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        session = Session()
        y1 = [x.__dict__ for x in
             session.query(ComputeJob).filter_by(name=job_name).all()
             ][0]
        self.assertEqual(mgmt["job_name"], y1["name"])
        self.assertEqual(EStatus.RUN, y1["status"])

        update_job_status(self.config, status=EStatus.DONE)
        y2 = [x.__dict__ for x in
             session.query(ComputeJob).filter_by(name=job_name).all()
             ][0]
        self.assertEqual(EStatus.DONE, y2["status"])

    def tearDown(self):
        os.remove("test.db")


if __name__ == '__main__':
    unittest.main()
